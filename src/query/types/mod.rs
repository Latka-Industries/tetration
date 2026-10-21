//! JSON query / response types.

mod cell_filter;
mod document;
mod error;
mod plan;
mod response;
mod transform_method;

pub(crate) use cell_filter::{CellPredicate, INTEGER_FILTER_FILL_MSG, fold_keeps_cell};

pub use document::{
    AlignSpec, AxisSlice, ExecutionDeviceHint, ExecutionHints, GroupBySpec, Operation, OutputHint,
    OutputHints, PipelineStep, QueryDocument, ValueCompareOp, ValueFilter, ValuePredicate,
    WindowMode, WindowSpec, WriteHints, WriteTarget,
};
pub use error::TetError;
pub(crate) use plan::LabelGroupRemap;
pub use plan::{CHUNK_TOUCH_POLICY, ChunkTouchPolicy, PlannedChunkIo, ReadPlan};
pub use response::{
    AlignmentAxisResult, AlignmentResult, DatasetResolution, QueryExecutionPreview, QueryResponse,
};
pub use transform_method::TransformMethod;

pub(crate) use response::{ExecutionPreviewIo, OperationPreviewFields, QueryExecutionPreviewBuild};
