//! Dimension name → decimal axis index resolution.

use crate::catalog::DatasetMetadataV1;
use crate::query::resolve_axes::resolve_query_document_axes;
use crate::query::types::{Operation, QueryDocument};

fn minimal_doc(operation: Operation) -> QueryDocument {
    QueryDocument {
        layout_version: None,
        dataset: "a".into(),
        selection: None,
        filter: None,
        group_by: None,
        operation: Some(operation),
        output: None,
        write: None,
        execution: None,
        pipeline: None,
    }
}

#[test]
fn resolve_time_to_zero() {
    let meta = DatasetMetadataV1 {
        dim_names: Some(vec!["time".into(), "lat".into()]),
        ..Default::default()
    };
    let mut doc = minimal_doc(Operation::Mean {
        axes: vec!["time".into()],
    });
    resolve_query_document_axes(&mut doc, Some(&meta), 2).unwrap();
    assert_eq!(doc.operation.as_ref().unwrap().axes(), &["0"]);
}

#[test]
fn unknown_name_errors() {
    let meta = DatasetMetadataV1 {
        dim_names: Some(vec!["x".into()]),
        ..Default::default()
    };
    let mut doc = minimal_doc(Operation::Sum {
        axes: vec!["y".into()],
    });
    let err = resolve_query_document_axes(&mut doc, Some(&meta), 1).unwrap_err();
    assert!(err.to_string().contains("unknown dimension name"));
}
