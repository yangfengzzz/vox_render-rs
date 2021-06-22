/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::*;
use crate::track::*;
use crate::quaternion::Quaternion;

// TrackSamplingJob internal implementation. See *TrackSamplingJob for more
// details.
pub struct TrackSamplingJob<'a, _Track, ValueType> {
    // Ratio used to sample track, clamped in range [0,1] before job execution. 0
    // is the beginning of the track, 1 is the end. This is a ratio rather than a
    // ratio because tracks have no duration.
    pub ratio: f32,

    // Track to sample.
    pub track: Option<&'a _Track>,

    // Job output.
    pub result: ValueType,
}

impl<'a, ValueType: FloatType + FloatType<ImplType=ValueType>> TrackSamplingJob<'a, Track<ValueType>, ValueType> {
    pub fn new() -> TrackSamplingJob<'a, Track<ValueType>, ValueType> {
        return TrackSamplingJob {
            ratio: 0.0,
            track: None,
            result: ValueType::new_default(),
        };
    }

    // Validates all parameters.
    pub fn validate(&self) -> bool {
        let mut success = true;
        success &= self.track.is_some();
        return success;
    }

    // Validates and executes sampling.
    pub fn run(&mut self) -> bool {
        if !self.validate() {
            return false;
        }

        // Clamps ratio in range [0,1].
        let clamped_ratio = self.ratio.clamp(0.0, 1.0);

        // Search keyframes to interpolate.
        let ratios = self.track.as_ref().unwrap().ratios();
        let values = self.track.as_ref().unwrap().values();
        debug_assert!(ratios.len() == values.len() && self.track.as_ref().unwrap().steps().len() * 8 >= values.len());

        // Default track returns identity.
        if ratios.len() == 0 {
            self.result = TrackPolicy::<ValueType>::identity();
            return true;
        }

        // Search for the first key frame with a ratio value greater than input ratio.
        // Our ratio is between this one and the previous one.
        let ptk1 = ratios.partition_point(|p| {
            p > &clamped_ratio
        });

        // Deduce keys indices.
        let id1 = ptk1;
        let id0 = id1 - 1;

        let id0step = (self.track.as_ref().unwrap().steps()[id0 / 8] & (1 << (id0 & 7))) != 0;
        if id0step || ptk1 == ratios.len() {
            self.result = values[id0].clone();
        } else {
            // Lerp relevant keys.
            let tk0 = ratios[id0];
            let tk1 = ratios[id1];
            debug_assert!(clamped_ratio >= tk0 && clamped_ratio < tk1 && tk0 != tk1);
            let alpha = (clamped_ratio - tk0) / (tk1 - tk0);
            let vk0 = &values[id0];
            let vk1 = &values[id1];
            self.result = TrackPolicy::<ValueType>::lerp(vk0, vk1, alpha);
        }
        return true;
    }
}

impl<'a> TrackSamplingJob<'a, QuaternionTrack, Quaternion> {
    pub fn new() -> TrackSamplingJob<'a, QuaternionTrack, Quaternion> {
        return TrackSamplingJob {
            ratio: 0.0,
            track: None,
            result: Quaternion::new_default(),
        };
    }

    // Validates all parameters.
    pub fn validate(&self) -> bool {
        let mut success = true;
        success &= self.track.is_some();
        return success;
    }

    // Validates and executes sampling.
    pub fn run(&mut self) -> bool {
        if !self.validate() {
            return false;
        }

        // Clamps ratio in range [0,1].
        let clamped_ratio = self.ratio.clamp(0.0, 1.0);

        // Search keyframes to interpolate.
        let ratios = self.track.as_ref().unwrap().ratios();
        let values = self.track.as_ref().unwrap().values();
        debug_assert!(ratios.len() == values.len() && self.track.as_ref().unwrap().steps().len() * 8 >= values.len());

        // Default track returns identity.
        if ratios.len() == 0 {
            self.result = TrackPolicy::<Quaternion>::identity();
            return true;
        }

        // Search for the first key frame with a ratio value greater than input ratio.
        // Our ratio is between this one and the previous one.
        let ptk1 = ratios.partition_point(|p| {
            p > &clamped_ratio
        });

        // Deduce keys indices.
        let id1 = ptk1;
        let id0 = id1 - 1;

        let id0step = (self.track.as_ref().unwrap().steps()[id0 / 8] & (1 << (id0 & 7))) != 0;
        if id0step || ptk1 == ratios.len() {
            self.result = values[id0].clone();
        } else {
            // Lerp relevant keys.
            let tk0 = ratios[id0];
            let tk1 = ratios[id1];
            debug_assert!(clamped_ratio >= tk0 && clamped_ratio < tk1 && tk0 != tk1);
            let alpha = (clamped_ratio - tk0) / (tk1 - tk0);
            let vk0 = &values[id0];
            let vk1 = &values[id1];
            self.result = TrackPolicy::<Quaternion>::lerp(vk0, vk1, alpha);
        }
        return true;
    }
}

// Track sampling job implementation. Track sampling allows to query a track
// value for a specified ratio. This is a ratio rather than a time because
// tracks have no duration.
pub type FloatTrackSamplingJob<'a> = TrackSamplingJob<'a, FloatTrack, Float>;
pub type Float2TrackSamplingJob<'a> = TrackSamplingJob<'a, Float2Track, Float2>;
pub type Float3TrackSamplingJob<'a> = TrackSamplingJob<'a, Float3Track, Float3>;
pub type Float4TrackSamplingJob<'a> = TrackSamplingJob<'a, Float4Track, Float4>;
pub type QuaternionTrackSamplingJob<'a> = TrackSamplingJob<'a, QuaternionTrack, Quaternion>;

#[cfg(test)]
mod track_sampling_job {
    use crate::raw_track::RawFloatTrack;
    use crate::track_builder::TrackBuilder;
    use crate::track_sampling_job::FloatTrackSamplingJob;

    #[test]
    fn job_validity() {
        // Building default RawFloatTrack succeeds.
        let raw_float_track = RawFloatTrack::new();
        assert_eq!(raw_float_track.validate(), true);

        // Builds track
        let track = TrackBuilder::apply_float(&raw_float_track);
        assert_eq!(track.is_some(), true);

        {  // Empty/default job
            let mut job = FloatTrackSamplingJob::new();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid track.
            let mut job = FloatTrackSamplingJob::new();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Valid
            let mut job = FloatTrackSamplingJob::new();
            job.track = track.as_ref();
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }

    #[test]
    fn default() {
        todo!()
    }

    #[test]
    fn bounds() {
        todo!()
    }

    #[test]
    fn float() {
        todo!()
    }

    #[test]
    fn float2() {
        todo!()
    }

    #[test]
    fn float3() {
        todo!()
    }

    #[test]
    fn float4() {
        todo!()
    }

    #[test]
    fn quaternion() {
        todo!()
    }
}