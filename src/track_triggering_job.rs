/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::track::FloatTrack;

// Structure of an edge as detected by the job.
pub struct Edge {
    // Ratio at which track value crossed threshold.
    pub ratio: f32,
    // true is edge is rising (getting higher than threshold).
    pub rising: bool,
}

// Track edge triggering job implementation. Edge triggering wording refers to
// signal processing, where a signal edge is a transition from low to high or
// from high to low. It is called an "edge" because of the square wave which
// represents a signal has edges at those points. A rising edge is the
// transition from low to high, a falling edge is from high to low.
// TrackTriggeringJob detects when track curve crosses a threshold value,
// triggering dated events that can be processed as state changes.
// Only FloatTrack is supported, because comparing to a threshold for other
// track types isn't possible.
// The job execution actually performs a lazy evaluation of edges. It builds an
// iterator that will process the next edge on each call to ++ operator.
pub struct TrackTriggeringJob<'a> {
    // Input range. 0 is the beginning of the track, 1 is the end.
    // from and to can be of any sign, any order, and any range. The job will
    // perform accordingly:
    // - if difference between from and to is greater than 1, the iterator will
    // loop multiple times on the track.
    // - if from is greater than to, then the track is processed backward (rising
    // edges in forward become falling ones).
    pub from: f32,
    pub to: f32,

    // Edge detection threshold value.
    // A rising edge is detected as soon as the track value becomes greater than
    // the threshold.
    // A falling edge is detected as soon as the track value becomes smaller or
    // equal than the threshold.
    pub threshold: f32,

    // Track to sample.
    pub track: Option<&'a FloatTrack>,
}

impl<'a> TrackTriggeringJob<'a> {
    pub fn new() -> TrackTriggeringJob<'a> {
        return TrackTriggeringJob {
            from: 0.0,
            to: 0.0,
            threshold: 0.0,
            track: None,
        };
    }

    // Validates job parameters.
    pub fn validate(&self) -> bool {
        let mut valid = true;
        valid &= self.track.is_some();
        return valid;
    }

    // Validates and executes job. Execution is lazy. Iterator operator ++ is
    // actually doing the processing work.
    pub fn run() -> bool {
        todo!()
    }
}