/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::Float3;
use crate::quaternion::Quaternion;
use crate::raw_animation::*;
use crate::transform::Transform;

// Translation interpolation method.
// This must be the same Lerp as the one used by the sampling job.
pub fn lerp_translation(_a: &Float3, _b: &Float3, _alpha: f32) -> Float3 {
    return Float3::lerp(_a, _b, _alpha);
}

// Rotation interpolation method.
// This must be the same Lerp as the one used by the sampling job.
// The goal is to take the shortest path between _a and _b. This code replicates
// this behavior that is actually not done at runtime, but when building the
// animation.
pub fn lerp_rotation(_a: &Quaternion, _b: &Quaternion, _alpha: f32) -> Quaternion {
    // Finds the shortest path. This is done by the AnimationBuilder for runtime
    // animations.
    let dot = _a.x * _b.x + _a.y * _b.y + _a.z * _b.z + _a.w * _b.w;
    return _a.nlerp(&match dot < 0.0 {
        true => -_b,
        false => *_b,
    }, _alpha);  // _b an -_b are the same rotation.
}

// Scale interpolation method.
pub fn lerp_scale(_a: &Float3, _b: &Float3, _alpha: f32) -> Float3 {
    return Float3::lerp(_a, _b, _alpha);
}

// Samples a RawAnimation track. This function shall be used for offline
// purpose. Use ozz::animation::Animation and ozz::animation::SamplingJob for
// runtime purpose.
// Returns false if track is invalid.
pub fn sample_track(_track: &JointTrack, _time: f32, _transform: &mut Transform) -> bool {
    if !_track.validate(f32::INFINITY) {
        return false;
    }

    sample_track_no_validate(_track, _time, _transform);
    return true;
}

// Samples a RawAnimation. This function shall be used for offline
// purpose. Use ozz::animation::Animation and ozz::animation::SamplingJob for
// runtime purpose.
// _animation must be valid.
// Returns false output range is too small or animation is invalid.
pub fn sample_animation(_animation: &RawAnimation, _time: f32, _transforms: &mut [Transform]) -> bool {
    if !_animation.validate() {
        return false;
    }
    if _animation.tracks.len() > _transforms.len() {
        return false;
    }

    for i in 0.._animation.tracks.len() {
        sample_track_no_validate(&_animation.tracks[i], _time, &mut _transforms[i]);
    }
    return true;
}

pub fn sample_component<T, _Key: KeyType<T>>(_track: &Vec<_Key>,
                                             _lerp: fn(_a: &T, _b: &T, _alpha: f32) -> T,
                                             _time: f32) -> T {
    return if _track.len() == 0 {
        // Return identity if there's no key for this track.
        _Key::identity()
    } else if _time <= _track.first().unwrap().time() {
        // Returns the first keyframe if _time is before the first keyframe.
        _track.first().unwrap().value()
    } else if _time >= _track.last().unwrap().time() {
        // Returns the last keyframe if _time is before the last keyframe.
        _track.last().unwrap().value()
    } else {
        // Needs to interpolate the 2 keyframes before and after _time.
        debug_assert!(_track.len() >= 2);
        let index = _track.partition_point(|ele| {
            return ele.time() < _time;
        });
        debug_assert!(_time >= _track[index - 1].time() && _time <= _track[index].time());
        let alpha = (_time - _track[index - 1].time()) / (_track[index].time() - _track[index - 1].time());
        _lerp(&_track[index - 1].value(), &_track[index].value(), alpha)
    };
}

fn sample_track_no_validate(_track: &JointTrack, _time: f32, _transform: &mut Transform) {
    _transform.translation = sample_component(&_track.translations, lerp_translation, _time);
    _transform.rotation = sample_component(&_track.rotations, lerp_rotation, _time);
    _transform.scale = sample_component(&_track.scales, lerp_scale, _time);
}

//--------------------------------------------------------------------------------------------------
// Implement fixed rate keyframe time iteration. This utility purpose is to
// ensure that sampling goes strictly from 0 to duration, and that period
// between consecutive time samples have a fixed period.
// This sounds trivial, but floating point error could occur if keyframe time
// was accumulated for a long duration.
pub struct FixedRateSamplingTime {
    duration_: f32,
    period_: f32,
    num_keys_: usize,
}

impl FixedRateSamplingTime {
    pub fn new(_duration: f32, _frequency: f32) -> FixedRateSamplingTime {
        return FixedRateSamplingTime {
            duration_: _duration,
            period_: 1.0 / _frequency,
            num_keys_: f32::ceil(1.0 + _duration * _frequency) as usize,
        };
    }

    pub fn time(&self, _key: usize) -> f32 {
        debug_assert!(_key < self.num_keys_);
        return f32::min(_key as f32 * self.period_, self.duration_);
    }

    pub fn num_keys(&self) -> usize { return self.num_keys_; }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod utils {
    use crate::raw_animation::JointTrack;
    use crate::transform::Transform;
    use crate::raw_animation_utils::*;
    use crate::math_test_helper::*;
    use crate::*;

    #[test]
    fn sampling_track_empty() {
        let track = JointTrack {
            translations: vec![],
            rotations: vec![],
            scales: vec![],
        };
        let mut output = Transform::identity();

        assert_eq!(sample_track(&track, 0.0, &mut output), true);

        expect_float3_eq!(output.translation, 0.0, 0.0, 0.0);
        expect_quaternion_eq!(output.rotation, 0.0, 0.0, 0.0, 1.0);
        expect_float3_eq!(output.scale, 1.0, 1.0, 1.0);
    }

    #[test]
    fn sampling_track_invalid() {
        // Key order
        {
            let mut track = JointTrack {
                translations: vec![],
                rotations: vec![],
                scales: vec![],
            };

            let t0 = TranslationKey { time: 0.9, value: Float3::new(1.0, 2.0, 4.0) };
            track.translations.push(t0);
            let t1 = TranslationKey { time: 0.1, value: Float3::new(2.0, 4.0, 8.0) };
            track.translations.push(t1);

            let mut output = Transform::identity();
            assert_eq!(sample_track(&track, 0.0, &mut output), false);
        }

        // Negative time
        {
            let mut track = JointTrack {
                translations: vec![],
                rotations: vec![],
                scales: vec![],
            };

            let t0 = TranslationKey { time: -1.0, value: Float3::new(1.0, 2.0, 4.0) };
            track.translations.push(t0);

            let mut output = Transform::identity();
            assert_eq!(sample_track(&track, 0.0, &mut output), false);
        }
    }

    #[test]
    fn sampling_track() {
        let mut track = JointTrack {
            translations: vec![],
            rotations: vec![],
            scales: vec![],
        };

        let t0 = TranslationKey { time: 0.1, value: Float3::new(1.0, 2.0, 4.0) };
        track.translations.push(t0);
        let t1 = TranslationKey { time: 0.9, value: Float3::new(2.0, 4.0, 8.0) };
        track.translations.push(t1);

        let r0 = RotationKey {
            time: 0.0,
            value: Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677),
        };
        track.rotations.push(r0);
        let r1 = RotationKey {
            // /!\ Negated (other hemisphepre) quaternion
            time: 0.5,
            value: -Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677),
        };
        track.rotations.push(r1);
        let r2 = RotationKey {
            time: 1.0,
            value: Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677),
        };
        track.rotations.push(r2);

        let s0 = ScaleKey { time: 0.5, value: Float3::new(-1.0, -2.0, -4.0) };
        track.scales.push(s0);

        let mut output = Transform::identity();

        // t = -.1
        assert_eq!(sample_track(&track, -0.1, &mut output), true);
        expect_float3_eq!(output.translation, 1.0, 2.0, 4.0);
        expect_quaternion_eq!(output.rotation, 0.70710677, 0.0, 0.0, 0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t = 0
        assert_eq!(sample_track(&track, 0.0, &mut output), true);
        expect_float3_eq!(output.translation, 1.0, 2.0, 4.0);
        expect_quaternion_eq!(output.rotation, 0.70710677, 0.0, 0.0, 0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t = .1
        assert_eq!(sample_track(&track, 0.1, &mut output), true);
        expect_float3_eq!(output.translation, 1.0, 2.0, 4.0);
        expect_quaternion_eq!(output.rotation, 0.6172133, 0.1543033, 0.0, 0.7715167);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t = .4999999
        assert_eq!(sample_track(&track, 0.4999999, &mut output), true);
        expect_float3_eq!(output.translation, 1.5, 3.0, 6.0);
        expect_quaternion_eq!(output.rotation, 0.0, 0.70710677, 0.0, 0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t = .5
        assert_eq!(sample_track(&track, 0.5, &mut output), true);
        expect_float3_eq!(output.translation, 1.5, 3.0, 6.0);
        expect_quaternion_eq!(output.rotation, 0.0, 0.70710677, 0.0, 0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t = .75
        assert_eq!(sample_track(&track, 0.75, &mut output), true);
        // Fixed up based on dot with previous quaternion
        expect_quaternion_eq!(output.rotation, 0.0, -0.70710677, 0.0, -0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t= .9
        assert_eq!(sample_track(&track, 0.9, &mut output), true);
        expect_float3_eq!(output.translation, 2.0, 4.0, 8.0);
        expect_quaternion_eq!(output.rotation, 0.0, -0.70710677, 0.0, -0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t= 1.
        assert_eq!(sample_track(&track, 1.0, &mut output), true);
        expect_float3_eq!(output.translation, 2.0, 4.0, 8.0);
        expect_quaternion_eq!(output.rotation, 0.0, 0.70710677, 0.0, 0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);

        // t= 1.1
        assert_eq!(sample_track(&track, 1.1, &mut output), true);
        expect_float3_eq!(output.translation, 2.0, 4.0, 8.0);
        expect_quaternion_eq!(output.rotation, 0.0, 0.70710677, 0.0, 0.70710677);
        expect_float3_eq!(output.scale, -1.0, -2.0, -4.0);
    }

    #[test]
    fn sampling_animation() {
        // Building an Animation with unsorted keys fails.
        let mut raw_animation = RawAnimation::new();
        raw_animation.duration = 2.0;
        raw_animation.tracks.push(JointTrack {
            translations: vec![],
            rotations: vec![],
            scales: vec![],
        });
        raw_animation.tracks.push(JointTrack {
            translations: vec![],
            rotations: vec![],
            scales: vec![],
        });

        let a = TranslationKey { time: 0.2, value: Float3::new(-1.0, 0.0, 0.0) };
        raw_animation.tracks[0].translations.push(a);

        let b = TranslationKey { time: 0.0, value: Float3::new(2.0, 0.0, 0.0) };
        raw_animation.tracks[1].translations.push(b);
        let c = TranslationKey { time: 0.2, value: Float3::new(6.0, 0.0, 0.0) };
        raw_animation.tracks[1].translations.push(c);
        let d = TranslationKey { time: 0.4, value: Float3::new(8.0, 0.0, 0.0) };
        raw_animation.tracks[1].translations.push(d);

        let mut output = [Transform::identity(), Transform::identity()];

        // Too small
        {
            let mut small = [Transform::identity(); 1];
            assert_eq!(sample_animation(&raw_animation, 0.0, &mut small), false);
        }

        // Invalid, cos track are longer than duration
        {
            raw_animation.duration = 0.1;
            assert_eq!(sample_animation(&raw_animation, 0.0, &mut output), false);
            raw_animation.duration = 2.0;
        }

        assert_eq!(sample_animation(&raw_animation, -0.1, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_quaternion_eq!(output[0].rotation, 0.0, 0.0, 0.0, 1.0);
        expect_float3_eq!(output[0].scale, 1.0, 1.0, 1.0);
        expect_float3_eq!(output[1].translation, 2.0, 0.0, 0.0);
        expect_quaternion_eq!(output[1].rotation, 0.0, 0.0, 0.0, 1.0);
        expect_float3_eq!(output[1].scale, 1.0, 1.0, 1.0);

        assert_eq!(sample_animation(&raw_animation, 0.0, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_float3_eq!(output[1].translation, 2.0, 0.0, 0.0);

        assert_eq!(sample_animation(&raw_animation, 0.2, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_float3_eq!(output[1].translation, 6.0, 0.0, 0.0);

        assert_eq!(sample_animation(&raw_animation, 0.3, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_float3_eq!(output[1].translation, 7.0, 0.0, 0.0);

        assert_eq!(sample_animation(&raw_animation, 0.4, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_float3_eq!(output[1].translation, 8.0, 0.0, 0.0);

        assert_eq!(sample_animation(&raw_animation, 2.0, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_float3_eq!(output[1].translation, 8.0, 0.0, 0.0);

        assert_eq!(sample_animation(&raw_animation, 3.0, &mut output), true);
        expect_float3_eq!(output[0].translation, -1.0, 0.0, 0.0);
        expect_float3_eq!(output[1].translation, 8.0, 0.0, 0.0);
    }

    #[test]
    fn fixed_rate_sampling_time() {
        {  // From 0
            let it = FixedRateSamplingTime::new(1.0, 30.0);
            assert_eq!(it.num_keys(), 31);

            assert_eq!(it.time(0), 0.0);
            assert_eq!(it.time(1), 1.0 / 30.0);
            assert_eq!(it.time(2), 2.0 / 30.0);
            expect_near!(it.time(29), 29.0 / 30.0, f32::EPSILON);
            assert_eq!(it.time(30), 1.0);
            // EXPECT_ASSERTION(it.time(31), "_key < num_keys");
        }

        {  // Offset
            let it = FixedRateSamplingTime::new(3.0, 100.0);
            assert_eq!(it.num_keys(), 301);

            assert_eq!(it.time(0), 0.0);
            assert_eq!(it.time(1), 1.0 / 100.0);
            assert_eq!(it.time(2), 2.0 / 100.0);
            assert_eq!(it.time(299), 299.0 / 100.0);
            assert_eq!(it.time(300), 3.0);
        }

        {  // Ceil
            let it = FixedRateSamplingTime::new(1.001, 30.0);
            assert_eq!(it.num_keys(), 32);

            assert_eq!(it.time(0), 0.0);
            assert_eq!(it.time(1), 1.0 / 30.0);
            assert_eq!(it.time(2), 2.0 / 30.0);
            expect_near!(it.time(29), 29.0 / 30.0, f32::EPSILON);
            assert_eq!(it.time(30), 1.0);
            assert_eq!(it.time(31), 1.001);
        }

        {  // Long
            let it = FixedRateSamplingTime::new(1000.0, 30.0);
            assert_eq!(it.num_keys(), 30001);

            assert_eq!(it.time(0), 0.0);
            assert_eq!(it.time(1), 1.0 / 30.0);
            assert_eq!(it.time(2), 2.0 / 30.0);
            expect_near!(it.time(29999), 29999.0 / 30.0, 1.0e-4);
            assert_eq!(it.time(30000), 1000.0);
        }
    }
}