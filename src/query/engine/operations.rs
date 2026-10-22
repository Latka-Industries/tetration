use std::path::Path;

use crate::query::{
    device, dispatch,
    engine::{budget, spill_policy},
    fold, gpu,
    materialize::{self, stats::run_tier_c_operation},
    types,
};
use crate::utils::dtype::ElementDtype;

fn materialized_io(
    materialized: &materialize::MaterializedLogical,
) -> (u64, budget::MemoryStrategy) {
    match materialized {
        materialize::MaterializedLogical::F32 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::F64 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::I32 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::I64 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::U8 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::U16 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::I16 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::U32 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::U64 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | materialize::MaterializedLogical::F16 {
            total_bytes_read_from_disk,
            strategy,
            ..
        } => (*total_bytes_read_from_disk, *strategy),
    }
}

fn preview_from_bundle(
    total_bytes_read_from_disk: u64,
    previews: materialize::DecodePreviewBundle,
    operation: types::OperationPreviewFields,
    memory_strategy: Option<&'static str>,
    spill_f32_path: Option<String>,
    spill_f32_bytes: Option<u64>,
) -> types::QueryExecutionPreview {
    types::QueryExecutionPreview::assemble(types::QueryExecutionPreviewBuild {
        io: types::ExecutionPreviewIo {
            total_bytes_read_from_disk,
            memory_strategy,
            spill_f32_path,
            spill_f32_bytes,
        },
        previews,
        operation,
    })
}

fn stamp_device_route(preview: &mut types::QueryExecutionPreview, route: device::DeviceRoute) {
    device::attach_device_fields(preview, route);
}

fn attach_budget_fields(
    preview: &mut types::QueryExecutionPreview,
    budget: budget::ExecutionBudget,
    plan: &types::ReadPlan,
    dtype: ElementDtype,
    fold_policy: Option<fold::fold_policy::FoldIoPolicy>,
) {
    preview.memory_budget_bytes = Some(budget.memory_budget_bytes);
    preview.host_available_ram_bytes = budget.host_available_ram_bytes;
    preview.logical_selection_bytes = budget
        .logical_element_bytes(dtype, plan.logical_element_count)
        .ok();
    if dtype == ElementDtype::F32 {
        preview.logical_selection_f32_bytes = preview.logical_selection_bytes;
    }
    if let Some(policy) = fold_policy {
        preview.fold_parallel = Some(policy.parallel);
        preview.fold_workers = policy.fold_workers;
        preview.io_regime = Some(policy.io_regime.as_str());
        preview.fold_linear_scan = Some(policy.linear_scan);
    }
}

struct FoldPreviewInput<'a> {
    folded: fold::FoldPlanOutcome,
    strategy: budget::MemoryStrategy,
    budget: budget::ExecutionBudget,
    plan: &'a types::ReadPlan,
    dtype: ElementDtype,
    fold_policy: fold::fold_policy::FoldIoPolicy,
    device_route: device::DeviceRoute,
}

fn fold_outcome_to_preview(input: FoldPreviewInput<'_>) -> types::QueryExecutionPreview {
    let FoldPreviewInput {
        folded,
        strategy,
        budget,
        plan,
        dtype,
        fold_policy,
        device_route,
    } = input;
    let mut preview = preview_from_bundle(
        folded.total_bytes_read_from_disk,
        materialize::DecodePreviewBundle {
            f32: folded.f32_preview,
            f64: folded.f64_preview,
            i32: folded.i32_preview,
            i64: folded.i64_preview,
            f32_truncated: folded.f32_preview_truncated,
            f64_truncated: folded.f64_preview_truncated,
            i32_truncated: folded.i32_preview_truncated,
            i64_truncated: folded.i64_preview_truncated,
            u8: folded.u8_preview,
            u8_truncated: folded.u8_preview_truncated,
            u16: folded.u16_preview,
            u16_truncated: folded.u16_preview_truncated,
            i16: folded.i16_preview,
            i16_truncated: folded.i16_preview_truncated,
            u32: folded.u32_preview,
            u32_truncated: folded.u32_preview_truncated,
            u64: folded.u64_preview,
            u64_truncated: folded.u64_preview_truncated,
            f16: folded.f16_preview,
            f16_truncated: folded.f16_preview_truncated,
        },
        folded.operation,
        Some(strategy.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, budget, plan, dtype, Some(fold_policy));
    stamp_device_route(&mut preview, device_route);
    preview
}

struct GpuScalarFoldInput<'a> {
    mmap: &'a [u8],
    plan: &'a types::ReadPlan,
    max_preview: usize,
    kind: fold::reduction::ReductionKind,
    dtype: ElementDtype,
    policy: &'a fold::fold_policy::FoldIoPolicy,
    tet_path: Option<&'a Path>,
    execution: Option<&'a types::ExecutionHints>,
    op: &'a types::Operation,
}

fn try_gpu_scalar_fold_or_cpu(
    input: &GpuScalarFoldInput<'_>,
) -> Result<(fold::FoldPlanOutcome, device::DeviceRoute), types::TetError> {
    let GpuScalarFoldInput {
        mmap,
        plan,
        max_preview,
        kind,
        dtype,
        policy,
        tet_path,
        execution,
        op,
    } = *input;
    let route = device::resolve_device_route(execution, plan, dtype, Some(op));
    if (plan.cell_filter.is_some() || plan.label_eq.is_some()) && route.gpu_reduce {
        let folded = dispatch::scalar_fold(mmap, plan, max_preview, kind, dtype, policy, tet_path)?;
        return Ok((
            folded,
            device::DeviceRoute::cpu_fallback(route.requested, "cell_filter"),
        ));
    }
    if route.gpu_reduce && matches!(dtype, ElementDtype::F32 | ElementDtype::F16) {
        match gpu::try_scalar_gpu_fold(mmap, plan, max_preview, kind, route, dtype) {
            Ok(pair) => return Ok(pair),
            Err(reason) => {
                let folded =
                    dispatch::scalar_fold(mmap, plan, max_preview, kind, dtype, policy, tet_path)?;
                return Ok((
                    folded,
                    device::DeviceRoute::cpu_fallback(route.requested, reason),
                ));
            }
        }
    }
    let folded = dispatch::scalar_fold(mmap, plan, max_preview, kind, dtype, policy, tet_path)?;
    Ok((folded, route))
}

struct MaterializeRequiredInput<'a> {
    mmap: &'a [u8],
    plan: &'a types::ReadPlan,
    op: &'a types::Operation,
    max_preview: usize,
    budget: &'a budget::ExecutionBudget,
    allowlist: &'a spill_policy::SpillPathAllowlist,
    dtype: ElementDtype,
    execution: Option<&'a types::ExecutionHints>,
}

fn run_materialize_required_operation(
    input: &MaterializeRequiredInput<'_>,
) -> Result<types::QueryExecutionPreview, types::TetError> {
    let MaterializeRequiredInput {
        mmap,
        plan,
        op,
        max_preview,
        budget,
        allowlist,
        dtype,
        execution,
    } = *input;
    let materialized =
        materialize::materialize_logical_selection(mmap, plan, budget, allowlist, dtype)?;
    let bundle = materialize::preview_from_materialized(
        &materialized,
        plan.logical_element_count,
        max_preview,
    )?;
    let operation = run_tier_c_operation(&materialized, plan, op)?;
    let (total_bytes_read_from_disk, strategy) = materialized_io(&materialized);
    let mut preview = preview_from_bundle(
        total_bytes_read_from_disk,
        bundle,
        operation,
        Some(strategy.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, *budget, plan, dtype, None);
    let device_route = device::resolve_device_route(execution, plan, dtype, Some(op));
    stamp_device_route(&mut preview, device_route);
    Ok(preview)
}

fn scalar_reduction_kind(op: &types::Operation) -> Option<fold::reduction::ReductionKind> {
    op.axes()
        .is_empty()
        .then(|| fold::reduction::ReductionKind::from(op))
}

fn spill_requested(output: Option<&types::OutputHints>) -> Option<&str> {
    match output.and_then(|o| o.preferred.as_ref()) {
        Some(types::OutputHint::SpillArray { handle }) => Some(handle.as_str()),
        _ => None,
    }
}

pub(super) struct ExecutionPreviewInput<'a> {
    pub mmap: &'a [u8],
    pub plan: &'a types::ReadPlan,
    pub dtype: u32,
    pub source_dataset: &'a str,
    pub operation: Option<&'a types::Operation>,
    pub window: Option<&'a types::WindowSpec>,
    pub output: Option<&'a types::OutputHints>,
    pub write: Option<&'a types::WriteHints>,
    pub max_f32: usize,
    pub budget: budget::ExecutionBudget,
    pub execution: Option<&'a types::ExecutionHints>,
    pub spill_allowlist: Option<&'a spill_policy::SpillPathAllowlist>,
    pub tet_path: Option<&'a Path>,
}

pub(super) fn build_execution_preview(
    input: &ExecutionPreviewInput<'_>,
) -> Result<types::QueryExecutionPreview, types::TetError> {
    let ExecutionPreviewInput {
        mmap,
        plan,
        dtype,
        source_dataset,
        operation,
        window,
        output,
        write,
        max_f32,
        budget,
        execution,
        spill_allowlist,
        tet_path,
    } = *input;
    let elem_dtype = ElementDtype::from_wire(dtype)?;

    match operation {
        None => build_decode_preview(&DecodePreviewInput {
            mmap,
            plan,
            output,
            max_preview: max_f32,
            budget,
            spill_allowlist,
            dtype: elem_dtype,
            execution,
        }),
        Some(op) => build_operation_preview(&OperationPreviewInput {
            mmap,
            plan,
            op,
            window,
            write,
            source_dataset,
            max_preview: max_f32,
            budget,
            execution,
            spill_allowlist,
            dtype: elem_dtype,
            tet_path,
        }),
    }
}

struct OperationPreviewInput<'a> {
    mmap: &'a [u8],
    plan: &'a types::ReadPlan,
    op: &'a types::Operation,
    window: Option<&'a types::WindowSpec>,
    write: Option<&'a types::WriteHints>,
    source_dataset: &'a str,
    max_preview: usize,
    budget: budget::ExecutionBudget,
    execution: Option<&'a types::ExecutionHints>,
    spill_allowlist: Option<&'a spill_policy::SpillPathAllowlist>,
    dtype: ElementDtype,
    tet_path: Option<&'a Path>,
}

struct DecodePreviewInput<'a> {
    mmap: &'a [u8],
    plan: &'a types::ReadPlan,
    output: Option<&'a types::OutputHints>,
    max_preview: usize,
    budget: budget::ExecutionBudget,
    spill_allowlist: Option<&'a spill_policy::SpillPathAllowlist>,
    dtype: ElementDtype,
    execution: Option<&'a types::ExecutionHints>,
}

fn build_decode_preview(
    input: &DecodePreviewInput<'_>,
) -> Result<types::QueryExecutionPreview, types::TetError> {
    let DecodePreviewInput {
        mmap,
        plan,
        output,
        max_preview,
        budget,
        spill_allowlist,
        dtype,
        execution,
    } = *input;
    if let Some(spill_path) = spill_requested(output) {
        let path = Path::new(spill_path);
        let policy = spill_allowlist.ok_or_else(|| {
            types::TetError::Validation(
                "spill output requires a spill path allowlist (pass `--tet` so defaults apply)"
                    .into(),
            )
        })?;
        let resolved = policy.validate(path)?;
        let total_bytes_read_from_disk =
            dispatch::spill_full_selection(mmap, plan, &resolved, dtype)?;
        let bundle = dispatch::spill_export_preview(
            &resolved,
            plan.logical_element_count,
            max_preview,
            dtype,
        )?;
        let spill_bytes = budget.logical_element_bytes(dtype, plan.logical_element_count)?;
        let mut preview = preview_from_bundle(
            total_bytes_read_from_disk,
            bundle,
            types::OperationPreviewFields::default(),
            Some(budget::MemoryStrategy::MmapSpill.as_str()),
            Some(resolved.display().to_string()),
            Some(spill_bytes),
        );
        attach_budget_fields(&mut preview, budget, plan, dtype, None);
        stamp_device_route(
            &mut preview,
            device::resolve_device_route(execution, plan, dtype, None),
        );
        return Ok(preview);
    }
    if budget.full_tensor_exceeds_budget(plan, dtype)? && max_preview == 0 {
        return Err(types::TetError::Validation(format!(
            "logical selection ({} elements, {} bytes) exceeds memory_budget_bytes ({}); \
             use a positive preview cap, a reduction key, `spill`, or raise execution.memory_budget_bytes / memory_budget_percent_bps",
            plan.logical_element_count,
            budget.logical_element_bytes(dtype, plan.logical_element_count)?,
            budget.memory_budget_bytes
        )));
    }
    let (bundle, total_bytes) =
        dispatch::materialize_for_execution(mmap, plan, Some(max_preview), dtype)?;
    let mut preview = preview_from_bundle(
        total_bytes,
        bundle,
        types::OperationPreviewFields::default(),
        Some(budget::MemoryStrategy::CappedInMemory.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, budget, plan, dtype, None);
    stamp_device_route(
        &mut preview,
        device::resolve_device_route(execution, plan, dtype, None),
    );
    Ok(preview)
}

fn require_spill_allowlist<'a>(
    spill_allowlist: Option<&'a spill_policy::SpillPathAllowlist>,
    context: &str,
) -> Result<&'a spill_policy::SpillPathAllowlist, types::TetError> {
    spill_allowlist.ok_or_else(|| {
        types::TetError::Validation(format!(
            "{context} needs a spill path allowlist (pass `--tet` so defaults apply)"
        ))
    })
}

/// Run a two-pass transform and assemble the execution preview.
fn build_transform_operation_preview(
    input: &OperationPreviewInput<'_>,
) -> Result<types::QueryExecutionPreview, types::TetError> {
    let OperationPreviewInput {
        mmap,
        plan,
        op,
        write,
        source_dataset,
        max_preview,
        budget,
        execution,
        spill_allowlist,
        dtype,
        tet_path,
        ..
    } = *input;
    let policy = require_spill_allowlist(spill_allowlist, "transform operation")?;
    let fold_policy =
        fold::fold_policy::FoldIoPolicy::resolve(plan, &budget, execution, dtype, true)?;
    let transform_input = crate::query::transform::TransformRunInput {
        mmap,
        plan,
        op,
        write,
        source_dataset,
        max_preview,
        budget,
        dtype,
        spill_allowlist: policy,
        tet_path,
        fold_policy,
    };
    let outcome = crate::query::transform::run_transform(&transform_input)?;
    let mut preview = preview_from_bundle(
        outcome.total_bytes_read_from_disk,
        outcome.bundle,
        outcome.operation,
        Some(outcome.strategy.as_str()),
        outcome.spill_path,
        outcome.spill_bytes,
    );
    attach_budget_fields(&mut preview, budget, plan, dtype, Some(fold_policy));
    stamp_device_route(
        &mut preview,
        device::resolve_device_route(execution, plan, dtype, Some(op)),
    );
    Ok(preview)
}

fn build_fold_operation_preview(
    input: &OperationPreviewInput<'_>,
) -> Result<types::QueryExecutionPreview, types::TetError> {
    let OperationPreviewInput {
        mmap,
        plan,
        op,
        window,
        max_preview,
        budget,
        execution,
        dtype,
        tet_path,
        ..
    } = *input;
    // Window folds have no linear_scan path; keep parallel chunk fold when OOC.
    let allows_linear_scan = window.is_none();
    let fold_policy = fold::fold_policy::FoldIoPolicy::resolve(
        plan,
        &budget,
        execution,
        dtype,
        allows_linear_scan,
    )?;
    if let Some(window) = window {
        let kind = fold::reduction::ReductionKind::from(op);
        let folded =
            fold::window::fold_read_plan_window_operation(&fold::window::WindowFoldInput {
                mmap,
                plan,
                max_preview,
                kind,
                window,
                axis_labels: op.axes(),
                dtype,
                policy: &fold_policy,
            })?;
        let device_route = device::resolve_device_route(execution, plan, dtype, Some(op));
        let mut preview = fold_outcome_to_preview(FoldPreviewInput {
            folded,
            strategy: budget::MemoryStrategy::StreamingFold,
            budget,
            plan,
            dtype,
            fold_policy,
            device_route,
        });
        if let Ok(layout) = fold::window_geometry::window_axis_layout(plan, window, op.axes()) {
            preview.window_count = Some(layout.window_count);
        }
        return Ok(preview);
    }
    if let Some(kind) = scalar_reduction_kind(op) {
        let (folded, device_route) = try_gpu_scalar_fold_or_cpu(&GpuScalarFoldInput {
            mmap,
            plan,
            max_preview,
            kind,
            dtype,
            policy: &fold_policy,
            tet_path,
            execution,
            op,
        })?;
        return Ok(fold_outcome_to_preview(FoldPreviewInput {
            folded,
            strategy: budget::MemoryStrategy::StreamingFold,
            budget,
            plan,
            dtype,
            fold_policy,
            device_route,
        }));
    }
    let kind = fold::reduction::ReductionKind::from(op);
    let folded = dispatch::partial_fold(
        mmap,
        plan,
        max_preview,
        kind,
        op.axes(),
        dtype,
        &fold_policy,
    )?;
    let device_route = device::resolve_device_route(execution, plan, dtype, Some(op));
    Ok(fold_outcome_to_preview(FoldPreviewInput {
        folded,
        strategy: budget::MemoryStrategy::StreamingFold,
        budget,
        plan,
        dtype,
        fold_policy,
        device_route,
    }))
}

fn build_operation_preview(
    input: &OperationPreviewInput<'_>,
) -> Result<types::QueryExecutionPreview, types::TetError> {
    let OperationPreviewInput { op, .. } = *input;
    if op.requires_transform() {
        return build_transform_operation_preview(input);
    }
    if op.requires_materialize() {
        let OperationPreviewInput {
            mmap,
            plan,
            op,
            max_preview,
            budget,
            spill_allowlist,
            dtype,
            execution,
            ..
        } = *input;
        let policy = require_spill_allowlist(spill_allowlist, "materialize-required operation")?;
        return run_materialize_required_operation(&MaterializeRequiredInput {
            mmap,
            plan,
            op,
            max_preview,
            budget: &budget,
            allowlist: policy,
            dtype,
            execution,
        });
    }
    build_fold_operation_preview(input)
}
