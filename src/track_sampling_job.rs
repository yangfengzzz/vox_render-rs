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
    pub result: Option<&'a mut ValueType>,
}

impl<'a, ValueType: FloatType + FloatType<ImplType=ValueType>> TrackSamplingJob<'a, Track<ValueType>, ValueType> {
    pub fn new() -> TrackSamplingJob<'a, Track<ValueType>, ValueType> {
        return TrackSamplingJob {
            ratio: 0.0,
            track: None,
            result: None,
        };
    }

    // Validates all parameters.
    pub fn validate(&self) -> bool {
        let mut success = true;
        success &= self.result.is_some();
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
            **self.result.as_mut().unwrap() = TrackPolicy::<ValueType>::identity();
            return true;
        }

        // Search for the first key frame with a ratio value greater than input ratio.
        // Our ratio is between this one and the previous one.
        let ptk1 = ratios.partition_point(|p| {
            p <= &clamped_ratio
        });

        // Deduce keys indices.
        let id1 = ptk1;
        let id0 = id1 - 1;

        let id0step = (self.track.as_ref().unwrap().steps()[id0 / 8] & (1 << (id0 & 7))) != 0;
        if id0step || ptk1 == ratios.len() {
            **self.result.as_mut().unwrap() = values[id0].clone();
        } else {
            // Lerp relevant keys.
            let tk0 = ratios[id0];
            let tk1 = ratios[id1];
            debug_assert!(clamped_ratio >= tk0 && clamped_ratio < tk1 && tk0 != tk1);
            let alpha = (clamped_ratio - tk0) / (tk1 - tk0);
            let vk0 = &values[id0];
            let vk1 = &values[id1];
            **self.result.as_mut().unwrap() = TrackPolicy::<ValueType>::lerp(vk0, vk1, alpha);
        }
        return true;
    }
}

impl<'a> TrackSamplingJob<'a, QuaternionTrack, Quaternion> {
    pub fn new() -> TrackSamplingJob<'a, QuaternionTrack, Quaternion> {
        return TrackSamplingJob {
            ratio: 0.0,
            track: None,
            result: None,
        };
    }

    // Validates all parameters.
    pub fn validate(&self) -> bool {
        let mut success = true;
        success &= self.result.is_some();
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
            **self.result.as_mut().unwrap() = TrackPolicy::<Quaternion>::identity();
            return true;
        }

        // Search for the first key frame with a ratio value greater than input ratio.
        // Our ratio is between this one and the previous one.
        let ptk1 = ratios.partition_point(|p| {
            p <= &clamped_ratio
        });

        // Deduce keys indices.
        let id1 = ptk1;
        let id0 = id1 - 1;

        let id0step = (self.track.as_ref().unwrap().steps()[id0 / 8] & (1 << (id0 & 7))) != 0;
        if id0step || ptk1 == ratios.len() {
            **self.result.as_mut().unwrap() = values[id0].clone();
        } else {
            // Lerp relevant keys.
            let tk0 = ratios[id0];
            let tk1 = ratios[id1];
            debug_assert!(clamped_ratio >= tk0 && clamped_ratio < tk1 && tk0 != tk1);
            let alpha = (clamped_ratio - tk0) / (tk1 - tk0);
            let vk0 = &values[id0];
            let vk1 = &values[id1];
            **self.result.as_mut().unwrap() = TrackPolicy::<Quaternion>::lerp(vk0, vk1, alpha);
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

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod track_sampling_job {
    use crate::raw_track::*;
    use crate::track_builder::TrackBuilder;
    use crate::track_sampling_job::*;
    use crate::track::FloatTrack;
    use crate::quaternion::*;
    use crate::math_test_helper::*;
    use crate::*;

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

        {  // Invalid output
            let mut job = FloatTrackSamplingJob::new();
            job.track = track.as_ref();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid track.
            let mut job = FloatTrackSamplingJob::new();
            let mut result = Float::new_scalar(0.0);
            job.result = Some(&mut result);
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Valid
            let mut job = FloatTrackSamplingJob::new();
            job.track = track.as_ref();
            let mut result = Float::new_scalar(0.0);
            job.result = Some(&mut result);
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }

    #[test]
    fn default() {
        let default_track = FloatTrack::new();
        let mut job = FloatTrackSamplingJob::new();
        job.track = Some(&default_track);
        let mut result = Float::new_scalar(1.0);
        job.result = Some(&mut result);
        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);
        assert_eq!(job.result.as_ref().unwrap().x, 0.0);
    }

    #[test]
    fn bounds() {
        let mut result = Float::new_scalar(0.0);

        let mut raw_float_track = RawFloatTrack::new();

        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.0,
                                         Float::new_scalar(0.0));
        raw_float_track.keyframes.push(key0);
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KStep, 0.5,
                                         Float::new_scalar(46.0));
        raw_float_track.keyframes.push(key1);
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.7,
                                         Float::new_scalar(0.0));
        raw_float_track.keyframes.push(key2);

        // Builds track
        let track = TrackBuilder::apply_float(&raw_float_track);
        assert_eq!(track.is_some(), true);

        // Samples to verify build output.
        let mut sampling = FloatTrackSamplingJob::new();
        sampling.track = track.as_ref();
        sampling.result = Some(&mut result);

        sampling.ratio = 0.0 - 1.0e-7;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 1.0 + 1.0e-7;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 1.5;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
    }

    #[test]
    fn float() {
        let mut result = Float::new_scalar(0.0);

        let mut raw_track = RawFloatTrack::new();

        let key0 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.0, Float::new_scalar(0.0));
        raw_track.keyframes.push(key0);
        let key1 = RawTrackKeyframe::new(
            RawTrackInterpolation::KStep, 0.5, Float::new_scalar(4.6));
        raw_track.keyframes.push(key1);
        let key2 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.7, Float::new_scalar(9.2));
        raw_track.keyframes.push(key2);
        let key3 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.9, Float::new_scalar(0.0));
        raw_track.keyframes.push(key3);

        // Builds track
        let track = TrackBuilder::apply_float(&raw_track);
        assert_eq!(track.is_some(), true);

        // Samples to verify build output.
        let mut sampling = FloatTrackSamplingJob::new();
        sampling.track = track.as_ref();
        sampling.result = Some(&mut result);

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 0.25;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 2.3);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 4.6);

        sampling.ratio = 0.6;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 4.6);

        sampling.ratio = 0.7;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 9.2);

        sampling.ratio = 0.8;
        assert_eq!(sampling.run(), true);
        expect_near!(sampling.result.as_ref().unwrap().x, 4.6, f32::EPSILON * 100.0);

        sampling.ratio = 0.9;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
    }

    #[test]
    fn float2() {
        let mut result = Float2::new_scalar(0.0);

        let mut raw_track = RawFloat2Track::new();

        let key0 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.0, Float2::new(0.0, 0.0));
        raw_track.keyframes.push(key0);
        let key1 = RawTrackKeyframe::new(
            RawTrackInterpolation::KStep, 0.5, Float2::new(2.3, 4.6));
        raw_track.keyframes.push(key1);
        let key2 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.7, Float2::new(4.6, 9.2));
        raw_track.keyframes.push(key2);
        let key3 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.9, Float2::new(0.0, 0.0));
        raw_track.keyframes.push(key3);

        // Builds track
        let track = TrackBuilder::apply_float2(&raw_track);
        assert_eq!(track.is_some(), true);

        // Samples to verify build output.
        let mut sampling = Float2TrackSamplingJob::new();
        sampling.track = track.as_ref();
        sampling.result = Some(&mut result);

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0);

        sampling.ratio = 0.25;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 1.15, 2.3);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 2.3, 4.6);

        sampling.ratio = 0.6;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 2.3, 4.6);

        sampling.ratio = 0.7;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 4.6, 9.2);

        sampling.ratio = 0.8;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 2.3, 4.6);

        sampling.ratio = 0.9;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        expect_float2_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0);
    }

    #[test]
    fn float3() {
        let mut result = Float3::new_scalar(0.0);

        let mut raw_track = RawFloat3Track::new();

        let key0 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.0, Float3::new(0.0, 0.0, 0.0));
        raw_track.keyframes.push(key0);
        let key1 = RawTrackKeyframe::new(
            RawTrackInterpolation::KStep, 0.5, Float3::new(0.0, 2.3, 4.6));
        raw_track.keyframes.push(key1);
        let key2 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.7, Float3::new(0.0, 4.6, 9.2));
        raw_track.keyframes.push(key2);
        let key3 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.9, Float3::new(0.0, 0.0, 0.0));
        raw_track.keyframes.push(key3);

        // Builds track
        let track = TrackBuilder::apply_float3(&raw_track);
        assert_eq!(track.is_some(), true);

        // Samples to verify build output.
        let mut sampling = Float3TrackSamplingJob::new();
        sampling.track = track.as_ref();
        sampling.result = Some(&mut result);

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0);

        sampling.ratio = 0.25;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 1.15, 2.3);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 2.3, 4.6);

        sampling.ratio = 0.6;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 2.3, 4.6);

        sampling.ratio = 0.7;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 4.6, 9.2);

        sampling.ratio = 0.8;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 2.3, 4.6);

        sampling.ratio = 0.9;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0);
    }

    #[test]
    fn float4() {
        let mut result = Float4::new_scalar(0.0);

        let mut raw_track = RawFloat4Track::new();

        let key0 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.0,
            Float4::new(0.0, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key0);
        let key1 = RawTrackKeyframe::new(
            RawTrackInterpolation::KStep, 0.5,
            Float4::new(0.0, 2.3, 0.0, 4.6));
        raw_track.keyframes.push(key1);
        let key2 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.7,
            Float4::new(0.0, 4.6, 0.0, 9.2));
        raw_track.keyframes.push(key2);
        let key3 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.9,
            Float4::new(0.0, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key3);

        // Builds track
        let track = TrackBuilder::apply_float4(&raw_track);
        assert_eq!(track.is_some(), true);

        // Samples to verify build output.
        let mut sampling = Float4TrackSamplingJob::new();
        sampling.track = track.as_ref();
        sampling.result = Some(&mut result);

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0, 0.0);

        sampling.ratio = 0.25;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 1.15, 0.0, 2.3);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 2.3, 0.0, 4.6);

        sampling.ratio = 0.6;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 2.3, 0.0, 4.6);

        sampling.ratio = 0.7;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 4.6, 0.0, 9.2);

        sampling.ratio = 0.8;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 2.3, 0.0, 4.6);

        sampling.ratio = 0.9;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0, 0.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0, 0.0);
    }

    #[test]
    fn quaternion() {
        let mut result = Quaternion::new_default();

        let mut raw_track = RawQuaternionTrack::new();

        let key0 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.0,
            Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677));
        raw_track.keyframes.push(key0);
        let key1 = RawTrackKeyframe::new(
            RawTrackInterpolation::KStep, 0.5,
            Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677));
        raw_track.keyframes.push(key1);
        let key2 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.7,
            Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677));
        raw_track.keyframes.push(key2);
        let key3 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.9, Quaternion::identity());
        raw_track.keyframes.push(key3);

        // Builds track
        let track = TrackBuilder::apply_quaternion(&raw_track);
        assert_eq!(track.is_some(), true);
        // Samples to verify build output.
        let mut sampling = QuaternionTrackSamplingJob::new();
        sampling.track = track.as_ref();
        sampling.result = Some(&mut result);

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.70710677, 0.0, 0.0, 0.70710677);

        sampling.ratio = 0.1;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.61721331, 0.15430345, 0.0, 0.77151674);

        sampling.ratio = 0.4999999;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);

        sampling.ratio = 0.6;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);

        sampling.ratio = 0.7;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.70710677, 0.0, 0.0, 0.70710677);

        sampling.ratio = 0.8;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.38268333, 0.0, 0.0, 0.92387962);

        sampling.ratio = 0.9;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0, 1.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.0, 0.0, 1.0);
    }
}