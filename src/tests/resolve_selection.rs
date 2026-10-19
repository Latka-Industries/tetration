//! Coordinate label → numeric slice resolution.

use std::collections::BTreeMap;

use crate::catalog::{CoordAxisV1, DatasetMetadataV1};
use crate::query::resolve_selection::resolve_query_document_selection;
use crate::query::types::{AxisSlice, QueryDocument};

#[test]
fn resolve_row_labels_to_indices() {
    let meta = DatasetMetadataV1 {
        dim_names: Some(vec!["row".into(), "col".into()]),
        coords: Some({
            let mut m = BTreeMap::new();
            m.insert(
                "row".into(),
                CoordAxisV1 {
                    labels: vec!["r0".into(), "r1".into()],
                },
            );
            m
        }),
        ..Default::default()
    };
    let mut doc = QueryDocument {
        layout_version: None,
        dataset: "a".into(),
        selection: Some(vec![
            AxisSlice {
                start: None,
                stop: None,
                step: None,
                start_label: Some("r0".into()),
                stop_label: Some("r1".into()),
            },
            AxisSlice {
                start: Some(0),
                stop: Some(3),
                step: None,
                start_label: None,
                stop_label: None,
            },
        ]),
        filter: None,
        group_by: None,
        operation: None,
        output: None,
        write: None,
        execution: None,
    };
    resolve_query_document_selection(&mut doc, Some(&meta), &[2, 3]).unwrap();
    let sel = doc.selection.as_ref().unwrap();
    assert_eq!(sel[0].start, Some(0));
    assert_eq!(sel[0].stop, Some(1));
    assert!(sel[0].start_label.is_none());
}
