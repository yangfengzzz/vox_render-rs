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
        todo!()
    }

    #[test]
    fn optimize_steps() {
        todo!()
    }

    #[test]
    fn optimize_interpolate() {
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