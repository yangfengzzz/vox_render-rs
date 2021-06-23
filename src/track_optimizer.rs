/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_track::*;
use std::marker::PhantomData;
use crate::vec_float::FloatType;
use crate::track::TrackPolicy;
use crate::quaternion::Quaternion;
use crate::decimate::decimate;
use crate::animation_optimizer::DecimateType;

// TrackOptimizer is responsible for optimizing an offline raw track instance.
// Optimization is a keyframe reduction process. Redundant and interpolable
// keyframes (within a tolerance value) are removed from the track. Default
// optimization tolerances are set in order to favor quality over runtime
// performances and memory footprint.
pub struct TrackOptimizer {
    // Optimization tolerance.
    pub tolerance: f32,
}

impl TrackOptimizer {
    pub fn new() -> TrackOptimizer {
        return TrackOptimizer {
            tolerance: 1.0e-3,
        };
    }

    // Optimizes _input using *this parameters.
    // Returns true on success and fills _output track with the optimized
    // version of _input track.
    // *_output must be a valid Raw*Track instance.
    // Returns false on failure and resets _output to an empty track.
    // See Raw*Track::Validate() for more details about failure reasons.
    pub fn apply_float(&self, _input: &RawFloatTrack, _output: &mut RawFloatTrack) -> bool {
        return optimize(self.tolerance, _input, _output);
    }
    pub fn apply_float2(&self, _input: &RawFloat2Track, _output: &mut RawFloat2Track) -> bool {
        return optimize(self.tolerance, _input, _output);
    }
    pub fn apply_float3(&self, _input: &RawFloat3Track, _output: &mut RawFloat3Track) -> bool {
        return optimize(self.tolerance, _input, _output);
    }
    pub fn apply_float4(&self, _input: &RawFloat4Track, _output: &mut RawFloat4Track) -> bool {
        return optimize(self.tolerance, _input, _output);
    }
    pub fn apply_quat(&self, _input: &RawQuaternionTrack, _output: &mut RawQuaternionTrack) -> bool {
        return optimize_quat(self.tolerance, _input, _output);
    }
}

struct Adapter<ValueType> {
    mark: PhantomData<ValueType>,
}

impl<ValueType: FloatType + FloatType<ImplType=ValueType>> DecimateType<RawTrackKeyframe<ValueType>> for Adapter<ValueType> {
    fn decimable(&self, _key: &RawTrackKeyframe<ValueType>) -> bool {
        // RawTrackInterpolation::kStep keyframes aren't optimized, as steps can't
        // be interpolated.
        return !matches!(_key.interpolation, RawTrackInterpolation::KStep);
    }

    fn lerp(&self, _left: &RawTrackKeyframe<ValueType>, _right: &RawTrackKeyframe<ValueType>,
            _ref: &RawTrackKeyframe<ValueType>) -> RawTrackKeyframe<ValueType> {
        debug_assert!(self.decimable(_ref));
        let alpha =
            (_ref.ratio - _left.ratio) / (_right.ratio - _left.ratio);
        debug_assert!(alpha >= 0.0 && alpha <= 1.0);
        let key = RawTrackKeyframe::new(_ref.interpolation.clone(), _ref.ratio,
                                        TrackPolicy::<ValueType>::lerp(&_left.value, &_right.value, alpha));
        return key;
    }

    fn distance(&self, _a: &RawTrackKeyframe<ValueType>, _b: &RawTrackKeyframe<ValueType>) -> f32 {
        return TrackPolicy::<ValueType>::distance(&_a.value, &_b.value);
    }
}

impl DecimateType<RawTrackKeyframe<Quaternion>> for Adapter<Quaternion> {
    fn decimable(&self, _key: &RawTrackKeyframe<Quaternion>) -> bool {
        // RawTrackInterpolation::kStep keyframes aren't optimized, as steps can't
        // be interpolated.
        return !matches!(_key.interpolation, RawTrackInterpolation::KStep);
    }

    fn lerp(&self, _left: &RawTrackKeyframe<Quaternion>, _right: &RawTrackKeyframe<Quaternion>,
            _ref: &RawTrackKeyframe<Quaternion>) -> RawTrackKeyframe<Quaternion> {
        debug_assert!(self.decimable(_ref));
        let alpha =
            (_ref.ratio - _left.ratio) / (_right.ratio - _left.ratio);
        debug_assert!(alpha >= 0.0 && alpha <= 1.0);
        let key = RawTrackKeyframe::new(_ref.interpolation.clone(), _ref.ratio,
                                        TrackPolicy::<Quaternion>::lerp(&_left.value, &_right.value, alpha));
        return key;
    }

    fn distance(&self, _a: &RawTrackKeyframe<Quaternion>, _b: &RawTrackKeyframe<Quaternion>) -> f32 {
        return TrackPolicy::<Quaternion>::distance(&_a.value, &_b.value);
    }
}

#[inline]
fn optimize<ValueType: FloatType + FloatType<ImplType=ValueType>>(
    _tolerance: f32, _input: &RawTrack<ValueType>, _output: &mut RawTrack<ValueType>) -> bool {
    // Reset output animation to default.
    *_output = RawTrack::new();

    // Validate animation.
    if !_input.validate() {
        return false;
    }

    // Copy name
    _output.name = _input.name.clone();

    // Optimizes.
    decimate(&_input.keyframes, &Adapter { mark: Default::default() }, _tolerance, &mut _output.keyframes);

    // Output animation is always valid though.
    return _output.validate();
}

#[inline]
fn optimize_quat(
    _tolerance: f32, _input: &RawTrack<Quaternion>, _output: &mut RawTrack<Quaternion>) -> bool {
    // Reset output animation to default.
    *_output = RawTrack::new();

    // Validate animation.
    if !_input.validate() {
        return false;
    }

    // Copy name
    _output.name = _input.name.clone();

    // Optimizes.
    decimate(&_input.keyframes, &Adapter { mark: Default::default() }, _tolerance, &mut _output.keyframes);

    // Output animation is always valid though.
    return _output.validate();
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod track_optimizer {
    use crate::track_optimizer::TrackOptimizer;
    use crate::raw_track::*;
    use crate::vec_float::*;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::quaternion::Quaternion;

    #[test]
    fn error() {
        let optimizer = TrackOptimizer::new();

        {  // Invalid input animation.
            let mut input = RawFloatTrack::new();
            input.keyframes.resize(1, RawTrackKeyframe {
                interpolation: RawTrackInterpolation::KStep,
                ratio: 0.0,
                value: Float::new_default(),
            });
            input.keyframes[0].ratio = 99.0;
            assert_eq!(input.validate(), false);

            // Builds animation
            let mut output = RawFloatTrack::new();
            output.keyframes.resize(1, RawTrackKeyframe {
                interpolation: RawTrackInterpolation::KStep,
                ratio: 0.0,
                value: Float::new_default(),
            });
            assert_eq!(optimizer.apply_float(&input, &mut output), false);
            assert_eq!(output.keyframes.len(), 0);
        }
    }

    #[test]
    fn name() {
        // Step keys aren't optimized.
        let optimizer = TrackOptimizer::new();

        let mut raw_float_track = RawFloatTrack::new();
        raw_float_track.name = "FloatTrackOptimizer test".to_string();

        let mut output = RawFloatTrack::new();
        assert_eq!(optimizer.apply_float(&raw_float_track, &mut output), true);

        assert_eq!(raw_float_track.name, output.name);
    }

    #[test]
    fn optimize_steps() {
        // Step keys aren't optimized.
        let optimizer = TrackOptimizer::new();

        let mut raw_float_track = RawFloatTrack::new();
        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KStep, 0.5,
                                         Float::new_scalar(46.0));
        raw_float_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KStep, 0.7,
                                         Float::new_scalar(0.0));
        raw_float_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KStep, 0.8,
                                         Float::new_scalar(1e-9));
        raw_float_track.keyframes.push(key2.clone());

        let mut output = RawFloatTrack::new();
        assert_eq!(optimizer.apply_float(&raw_float_track, &mut output), true);

        assert_eq!(output.keyframes.len(), 3);
        assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
        assert_eq!(output.keyframes[0].ratio, key0.ratio);
        assert_eq!(output.keyframes[0].value.x, key0.value.x);

        assert_eq!(output.keyframes[1].interpolation, key1.interpolation);
        assert_eq!(output.keyframes[1].ratio, key1.ratio);
        assert_eq!(output.keyframes[1].value.x, key1.value.x);

        assert_eq!(output.keyframes[2].interpolation, key2.interpolation);
        assert_eq!(output.keyframes[2].ratio, key2.ratio);
        assert_eq!(output.keyframes[2].value.x, key2.value.x);
    }

    #[test]
    fn optimize_interpolate() {
        // Step keys aren't optimized.
        let mut optimizer = TrackOptimizer::new();

        let mut raw_float_track = RawFloatTrack::new();
        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.0,
                                         Float::new_scalar(69.0));
        raw_float_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.25,
                                         Float::new_scalar(46.0));
        raw_float_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.5,
                                         Float::new_scalar(23.0));
        raw_float_track.keyframes.push(key2.clone());
        let key3 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                         0.500001, Float::new_scalar(23.000001));
        raw_float_track.keyframes.push(key3.clone());
        let key4 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.75,
                                         Float::new_scalar(0.0));
        raw_float_track.keyframes.push(key4.clone());
        let key5 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.8,
                                         Float::new_scalar(1e-12));
        raw_float_track.keyframes.push(key5.clone());
        let key6 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 1.0,
                                         Float::new_scalar(-1e-12));
        raw_float_track.keyframes.push(key6.clone());

        {
            let mut output = RawFloatTrack::new();

            optimizer.tolerance = 1e-3;
            assert_eq!(optimizer.apply_float(&raw_float_track, &mut output), true);

            assert_eq!(output.keyframes.len(), 2);

            assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
            assert_eq!(output.keyframes[0].ratio, key0.ratio);
            assert_eq!(output.keyframes[0].value.x, key0.value.x);

            assert_eq!(output.keyframes[1].interpolation, key4.interpolation);
            assert_eq!(output.keyframes[1].ratio, key4.ratio);
            assert_eq!(output.keyframes[1].value.x, key4.value.x);
        }

        {
            let mut output = RawFloatTrack::new();
            optimizer.tolerance = 1e-9;
            assert_eq!(optimizer.apply_float(&raw_float_track, &mut output), true);

            assert_eq!(output.keyframes.len(), 4);

            assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
            assert_eq!(output.keyframes[0].ratio, key0.ratio);
            assert_eq!(output.keyframes[0].value.x, key0.value.x);

            assert_eq!(output.keyframes[1].interpolation, key2.interpolation);
            assert_eq!(output.keyframes[1].ratio, key2.ratio);
            assert_eq!(output.keyframes[1].value.x, key2.value.x);

            assert_eq!(output.keyframes[2].interpolation, key3.interpolation);
            assert_eq!(output.keyframes[2].ratio, key3.ratio);
            assert_eq!(output.keyframes[2].value.x, key3.value.x);

            assert_eq!(output.keyframes[3].interpolation, key4.interpolation);
            assert_eq!(output.keyframes[3].ratio, key4.ratio);
            assert_eq!(output.keyframes[3].value.x, key4.value.x);
        }
    }

    #[test]
    fn float() {
        let optimizer = TrackOptimizer::new();

        let mut raw_track = RawFloatTrack::new();
        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.0,
                                         Float::new_scalar(6.9));
        raw_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.25,
                                         Float::new_scalar(4.6));
        raw_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.5,
                                         Float::new_scalar(2.3));
        raw_track.keyframes.push(key2.clone());
        let key3 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.500001,
                                         Float::new_scalar(2.300001));
        raw_track.keyframes.push(key3.clone());
        let key4 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.75,
                                         Float::new_scalar(0.0));
        raw_track.keyframes.push(key4.clone());
        let key5 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.8,
                                         Float::new_scalar(1e-12));
        raw_track.keyframes.push(key5.clone());
        let key6 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 1.0,
                                         Float::new_scalar(-1e-12));
        raw_track.keyframes.push(key6.clone());

        let mut output = RawFloatTrack::new();
        assert_eq!(optimizer.apply_float(&raw_track, &mut output), true);

        assert_eq!(output.keyframes.len(), 2);

        assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
        assert_eq!(output.keyframes[0].ratio, key0.ratio);
        assert_eq!(output.keyframes[0].value.x, key0.value.x);

        assert_eq!(output.keyframes[1].interpolation, key4.interpolation);
        assert_eq!(output.keyframes[1].ratio, key4.ratio);
        assert_eq!(output.keyframes[1].value.x, key4.value.x);
    }

    #[test]
    fn float2() {
        let optimizer = TrackOptimizer::new();

        let mut raw_track = RawFloat2Track::new();
        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.0,
                                         Float2::new(6.9, 0.0));
        raw_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.25,
                                         Float2::new(4.6, 0.0));
        raw_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.5,
                                         Float2::new(2.3, 0.0));
        raw_track.keyframes.push(key2.clone());
        let key3 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                         0.500001,
                                         Float2::new(2.3000001, 0.0));
        raw_track.keyframes.push(key3.clone());
        let key4 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.75,
                                         Float2::new(0.0, 0.0));
        raw_track.keyframes.push(key4.clone());
        let key5 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.8,
                                         Float2::new(0.0, 1e-12));
        raw_track.keyframes.push(key5.clone());
        let key6 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 1.0,
                                         Float2::new(-1e-12, 0.0));
        raw_track.keyframes.push(key6.clone());

        let mut output = RawFloat2Track::new();
        assert_eq!(optimizer.apply_float2(&raw_track, &mut output), true);

        assert_eq!(output.keyframes.len(), 2);

        assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
        assert_eq!(output.keyframes[0].ratio, key0.ratio);
        expect_float2_eq!(output.keyframes[0].value, key0.value.x, key0.value.y);

        assert_eq!(output.keyframes[1].interpolation, key4.interpolation);
        assert_eq!(output.keyframes[1].ratio, key4.ratio);
        expect_float2_eq!(output.keyframes[1].value, key4.value.x, key4.value.y);
    }

    #[test]
    fn float3() {
        let optimizer = TrackOptimizer::new();

        let mut raw_track = RawFloat3Track::new();
        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.0,
                                         Float3::new(6.9, 0.0, 0.0));
        raw_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.25,
                                         Float3::new(4.6, 0.0, 0.0));
        raw_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.5,
                                         Float3::new(2.3, 0.0, 0.0));
        raw_track.keyframes.push(key2.clone());
        let key3 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.500001,
                                         Float3::new(2.3000001, 0.0, 0.0));
        raw_track.keyframes.push(key3.clone());
        let key4 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.75,
                                         Float3::new(0.0, 0.0, 0.0));
        raw_track.keyframes.push(key4.clone());
        let key5 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.8,
                                         Float3::new(0.0, 0.0, 1e-12));
        raw_track.keyframes.push(key5.clone());
        let key6 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 1.0,
                                         Float3::new(0.0, -1e-12, 0.0));
        raw_track.keyframes.push(key6.clone());

        let mut output = RawFloat3Track::new();
        assert_eq!(optimizer.apply_float3(&raw_track, &mut output), true);

        assert_eq!(output.keyframes.len(), 2);

        assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
        assert_eq!(output.keyframes[0].ratio, key0.ratio);
        expect_float3_eq!(output.keyframes[0].value, key0.value.x, key0.value.y,
                         key0.value.z);

        assert_eq!(output.keyframes[1].interpolation, key4.interpolation);
        assert_eq!(output.keyframes[1].ratio, key4.ratio);
        expect_float3_eq!(output.keyframes[1].value, key4.value.x, key4.value.y,
                         key4.value.z);
    }

    #[test]
    fn float4() {
        let optimizer = TrackOptimizer::new();

        let mut raw_track = RawFloat4Track::new();
        let key0 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.0,
                                         Float4::new(6.9, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.25,
                                         Float4::new(4.6, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.5,
                                         Float4::new(2.3, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key2.clone());
        let key3 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.500001,
                                         Float4::new(2.3000001, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key3.clone());
        let key4 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.75,
                                         Float4::new(0.0, 0.0, 0.0, 0.0));
        raw_track.keyframes.push(key4.clone());
        let key5 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.8,
                                         Float4::new(0.0, 0.0, 0.0, 1e-12));
        raw_track.keyframes.push(key5.clone());
        let key6 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 1.0,
                                         Float4::new(0.0, 0.0, 0.0, -1e-12));
        raw_track.keyframes.push(key6.clone());

        let mut output = RawFloat4Track::new();
        assert_eq!(optimizer.apply_float4(&raw_track, &mut output), true);

        assert_eq!(output.keyframes.len(), 2);

        assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
        assert_eq!(output.keyframes[0].ratio, key0.ratio);
        expect_float4_eq!(output.keyframes[0].value, key0.value.x, key0.value.y,
                         key0.value.z, key0.value.w);

        assert_eq!(output.keyframes[1].interpolation, key4.interpolation);
        assert_eq!(output.keyframes[1].ratio, key4.ratio);
        expect_float4_eq!(output.keyframes[1].value, key4.value.x, key4.value.y,
                         key4.value.z, key0.value.w);
    }

    #[test]
    fn quaternion() {
        let optimizer = TrackOptimizer::new();

        let mut raw_track = RawQuaternionTrack::new();
        let key0 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.0,
            Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677));
        raw_track.keyframes.push(key0.clone());
        let key1 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.1,
            Quaternion::new(0.6172133, 0.1543033, 0.0, 0.7715167));  // NLerp
        raw_track.keyframes.push(key1.clone());
        let key2 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.5,
            Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677));
        raw_track.keyframes.push(key2.clone());
        let key3 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.500001,
            Quaternion::new(0.0, 0.70710676, 0.0, 0.70710678));
        raw_track.keyframes.push(key3.clone());
        let key4 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.75,
            Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677));
        raw_track.keyframes.push(key4.clone());
        let key5 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 0.8,
            Quaternion::new(-0.0, -0.70710677, -0.0, -0.70710677));
        raw_track.keyframes.push(key5.clone());
        let key6 = RawTrackKeyframe::new(
            RawTrackInterpolation::KLinear, 1.0,
            Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677));
        raw_track.keyframes.push(key6.clone());

        let mut output = RawQuaternionTrack::new();
        assert_eq!(optimizer.apply_quat(&raw_track, &mut output), true);

        assert_eq!(output.keyframes.len(), 2);

        assert_eq!(output.keyframes[0].interpolation, key0.interpolation);
        assert_eq!(output.keyframes[0].ratio, key0.ratio);
        expect_quaternion_eq!(output.keyframes[0].value, key0.value.x, key0.value.y,
                             key0.value.z, key0.value.w);

        assert_eq!(output.keyframes[1].interpolation, key2.interpolation);
        assert_eq!(output.keyframes[1].ratio, key2.ratio);
        expect_quaternion_eq!(output.keyframes[1].value, key2.value.x, key2.value.y,
                             key2.value.z, key2.value.w);
    }
}