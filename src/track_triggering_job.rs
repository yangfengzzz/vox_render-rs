/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::track::FloatTrack;

// Structure of an edge as detected by the job.
#[derive(Clone)]
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

    // Job output iterator.
    pub iterator: Option<Iter<'a>>,
}

impl<'a> TrackTriggeringJob<'a> {
    pub fn new() -> TrackTriggeringJob<'a> {
        return TrackTriggeringJob {
            from: 0.0,
            to: 0.0,
            threshold: 0.0,
            track: None,
            iterator: None,
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
    pub fn run(&'a mut self) -> bool {
        if !self.validate() {
            return false;
        }

        // Triggering can only happen in a valid range of ratio.
        if self.from == self.to {
            **self.iterator.as_mut().as_ref().unwrap() = self.end();
            return true;
        }

        **self.iterator.as_mut().as_ref().unwrap() = Iter::new_job(self);

        return true;
    }

    // Returns an iterator referring to the past-the-end element. It should only
    // be used to test if iterator loop reached the end (using operator !=), and
    // shall not be dereference.
    pub fn end(&'a self) -> Iter<'a> {
        return Iter::new_end(self);
    }
}

// Iterator implementation. Calls to next operator will compute the next edge. It
// should be compared (using operator !=) to job's end iterator to test if the
// last edge has been reached.
pub struct Iter<'a> {
    // Job this iterator works on.
    job_: Option<&'a TrackTriggeringJob<'a>>,

    // Current value of the outer loop, aka a ratio cursor between from and to.
    outer_: f32,

    // Current value of the inner loop, aka a key frame index.
    inner_: i64,

    // Latest evaluated edge.
    edge_: Edge,
}

impl<'a> Iter<'a> {
    pub fn new() -> Iter<'a> {
        return Iter {
            job_: None,
            outer_: 0.0,
            inner_: 0,
            edge_: Edge { ratio: 0.0, rising: false },
        };
    }

    fn new_job(_job: &'a TrackTriggeringJob<'a>) -> Iter<'a> {
        return Iter {
            job_: Some(_job),
            outer_: f32::floor(_job.from),
            inner_: match _job.from < _job.to {
                true => 0,
                false => (_job.track.as_ref().unwrap().ratios().len() - 1) as i64
            },
            edge_: Edge { ratio: 0.0, rising: false },
        };
    }

    fn new_end(_job: &'a TrackTriggeringJob<'a>) -> Iter<'a> {
        return Iter {
            job_: Some(_job),
            outer_: 0.0,
            inner_: -2,
            edge_: Edge { ratio: 0.0, rising: false },
        };
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        let ratios = self.job_.as_ref().unwrap().track.as_ref().unwrap().ratios();
        let num_keys = ratios.len() as i64;

        if self.job_.as_ref().unwrap().to > self.job_.as_ref().unwrap().from {
            while self.outer_ < self.job_.as_ref().unwrap().to {
                while self.inner_ < num_keys as i64 {
                    let i0 = match self.inner_ == 0 {
                        true => { num_keys - 1 }
                        false => { self.inner_ - 1 }
                    };
                    if detect_edge(i0, self.inner_, true, self.job_.as_ref().unwrap(), &mut self.edge_) {
                        self.edge_.ratio += self.outer_;  // Convert to global ratio space.
                        if self.edge_.ratio >= self.job_.as_ref().unwrap().from &&
                            (self.edge_.ratio < self.job_.as_ref().unwrap().to || self.job_.as_ref().unwrap().to >= 1.0 + self.outer_) {
                            self.inner_ += 1;
                            return Some(self.edge_.clone());  // Yield found edge.
                        }
                        // Won't find any further edge.
                        if ratios[self.inner_ as usize] + self.outer_ >= self.job_.as_ref().unwrap().to {
                            break;
                        }
                    }
                    self.inner_ += 1
                }
                self.inner_ = 0;  // Ready for next loop.
                self.outer_ += 1.
            }
        } else {
            while self.outer_ + 1.0 > self.job_.as_ref().unwrap().to {
                while self.inner_ >= 0 {
                    let i0 = match self.inner_ == 0 {
                        true => num_keys - 1,
                        false => self.inner_ - 1
                    };
                    if detect_edge(i0, self.inner_, false, self.job_.as_ref().unwrap(), &mut self.edge_) {
                        self.edge_.ratio += self.outer_;  // Convert to global ratio space.
                        if self.edge_.ratio >= self.job_.as_ref().unwrap().to &&
                            (self.edge_.ratio < self.job_.as_ref().unwrap().from || self.job_.as_ref().unwrap().from >= 1.0 + self.outer_) {
                            self.inner_ -= 1;
                            return Some(self.edge_.clone());  // Yield found edge.
                        }
                    }
                    // Won't find any further edge.
                    if ratios[self.inner_ as usize] + self.outer_ <= self.job_.as_ref().unwrap().to {
                        break;
                    }

                    self.inner_ -= 1;
                }
                self.inner_ = (ratios.len() - 1) as i64;  // Ready for next loop.
                self.outer_ -= 1.0
            }
        }

        return None;
    }
}

#[inline]
fn detect_edge(_i0: i64, _i1: i64, _forward: bool,
               _job: &TrackTriggeringJob, _edge: &mut Edge) -> bool {
    let values = _job.track.as_ref().unwrap().values();

    let vk0 = values[_i0 as usize];
    let vk1 = values[_i1 as usize];

    let mut detected = false;
    if vk0.x <= _job.threshold && vk1.x > _job.threshold {
        // Rising edge
        _edge.rising = _forward;
        detected = true;
    } else if vk0.x > _job.threshold && vk1.x <= _job.threshold {
        // Falling edge
        _edge.rising = !_forward;
        detected = true;
    }

    if detected {
        let ratios = _job.track.as_ref().unwrap().ratios();
        let steps = _job.track.as_ref().unwrap().steps();

        let step = (steps[_i0 as usize / 8] & (1 << (_i0 & 7))) != 0;
        if step {
            _edge.ratio = ratios[_i1 as usize];
        } else {
            debug_assert!(vk0.x != vk1.x);  // Won't divide by 0

            if _i1 == 0 {
                _edge.ratio = 0.0;
            } else {
                // Finds where the curve crosses threshold value.
                // This is the lerp equation, where we know the result and look for
                // alpha, aka un-lerp.
                let alpha = (_job.threshold - vk0.x) / (vk1.x - vk0.x);

                // Remaps to keyframes actual times.
                let tk0 = ratios[_i0 as usize];
                let tk1 = ratios[_i1 as usize];
                _edge.ratio = (tk1 - tk0) * alpha + tk0;
            }
        }
    }
    return detected;
}