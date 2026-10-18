use crate::query::types::{ReadPlan, TetError};

pub(crate) fn validate_read_plan_geometry(plan: &ReadPlan, out_len: usize) -> Result<(), TetError> {
    let ndim = plan.dataset_shape.len();
    if plan.chunk_shape.len() != ndim
        || plan.selection_box_start.len() != ndim
        || plan.selection_box_stop_exclusive.len() != ndim
        || plan.selection_step.len() != ndim
        || plan.logical_selection_shape.len() != ndim
    {
        return Err(TetError::Validation(
            "read_plan geometry fields have inconsistent rank".into(),
        ));
    }
    if out_len > plan.logical_element_count {
        return Err(TetError::Validation(format!(
            "output buffer length {out_len} exceeds read_plan.logical_element_count {}",
            plan.logical_element_count
        )));
    }
    Ok(())
}

pub(crate) fn validate_full_read_plan_buffer(
    plan: &ReadPlan,
    out_len: usize,
) -> Result<(), TetError> {
    validate_read_plan_geometry(plan, out_len)?;
    if out_len != plan.logical_element_count {
        return Err(TetError::Validation(format!(
            "output buffer length {out_len} != read_plan.logical_element_count {}",
            plan.logical_element_count
        )));
    }
    Ok(())
}
