/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_track::*;
use crate::track::*;
use crate::vec_float::FloatType;
use crate::quaternion::Quaternion;

// Defines the class responsible of building runtime track instances from
// offline tracks.The input raw track is first validated. Runtime conversion of
// a validated raw track cannot fail. Note that no optimization is performed on
// the data at all.
pub struct TrackBuilder {}

impl TrackBuilder {
    // Creates a Track based on _raw_track and *this builder parameters.
    // Returns a track instance on success, an empty unique_ptr on failure. See
    // Raw*Track::validate() for more details about failure reasons.
    // The track is returned as an unique_ptr as ownership is given back to the
    // caller.
    pub fn apply_float(_input: &RawFloatTrack) -> Option<FloatTrack> {
        return TrackBuilder::build(_input);
    }
    pub fn apply_float2(_input: &RawFloat2Track) -> Option<Float2Track> {
        return TrackBuilder::build(_input);
    }
    pub fn apply_float3(_input: &RawFloat3Track) -> Option<Float3Track> {
        return TrackBuilder::build(_input);
    }
    pub fn apply_float4(_input: &RawFloat4Track) -> Option<Float4Track> {
        return TrackBuilder::build(_input);
    }

    fn build<ValueType: FloatType + FloatType<ImplType=ValueType>>(
        _input: &RawTrack<ValueType>) -> Option<Track<ValueType>> {
        // Tests _raw_animation validity.
        if !_input.validate() {
            return None;
        }

        // Everything is fine, allocates and fills the animation.
        // Nothing can fail now.
        let mut track = Track::<ValueType>::new();

        // Copy data to temporary prepared data structure
        let mut keyframes: Vec<RawTrackKeyframe<ValueType>> = Vec::new();
        // Guessing a worst size to avoid re-alloc.
        let worst_size = _input.keyframes.len() * 2 +  // * 2 in case all keys are kStep
            2;                             // + 2 for first and last keys
        keyframes.reserve(worst_size);

        // Ensure there's a key frame at the start and end of the track (required for
        // sampling).
        patch_begin_end_keys(_input, &mut keyframes);

        // Allocates output track.
        track.allocate(keyframes.len());

        // Copy all keys to output.
        debug_assert!(keyframes.len() == track.ratios_.len() &&
            keyframes.len() == track.values_.len() &&
            keyframes.len() <= track.steps_.len() * 8);
        track.steps_.resize(track.steps_.len(), 0);
        for i in 0..keyframes.len() {
            let src_key = &keyframes[i];
            track.ratios_[i] = src_key.ratio.clone();
            track.values_[i] = src_key.value.clone();
            track.steps_[i / 8] |= ((matches!(src_key.interpolation, RawTrackInterpolation::KStep) as usize) << (i & 7)) as u8;
        }

        // Copy track's name.
        if !_input.name.is_empty() {
            track.name_ = _input.name.clone();
        }

        return Some(track);  // Success.
    }
}

impl TrackBuilder {
    pub fn apply_quaternion(_input: &RawQuaternionTrack) -> Option<QuaternionTrack> {
        return TrackBuilder::build_quat(_input);
    }

    fn build_quat(_input: &RawTrack<Quaternion>) -> Option<Track<Quaternion>> {
        // Tests _raw_animation validity.
        if !_input.validate() {
            return None;
        }

        // Everything is fine, allocates and fills the animation.
        // Nothing can fail now.
        let mut track = Track::<Quaternion>::new();

        // Copy data to temporary prepared data structure
        let mut keyframes: Vec<RawTrackKeyframe<Quaternion>> = Vec::new();
        // Guessing a worst size to avoid re-alloc.
        let worst_size = _input.keyframes.len() * 2 +  // * 2 in case all keys are kStep
            2;                             // + 2 for first and last keys
        keyframes.reserve(worst_size);

        // Ensure there's a key frame at the start and end of the track (required for
        // sampling).
        patch_begin_end_keys_quat(_input, &mut keyframes);

        // Fixup values, ex: successive opposite quaternions that would fail to take
        // the shortest path during the normalized-lerp.
        fixup(&mut keyframes);

        // Allocates output track.
        track.allocate(keyframes.len());

        // Copy all keys to output.
        debug_assert!(keyframes.len() == track.ratios_.len() &&
            keyframes.len() == track.values_.len() &&
            keyframes.len() <= track.steps_.len() * 8);
        track.steps_.resize(track.steps_.len(), 0);
        for i in 0..keyframes.len() {
            let src_key = &keyframes[i];
            track.ratios_[i] = src_key.ratio.clone();
            track.values_[i] = src_key.value.clone();
            track.steps_[i / 8] |= ((matches!(src_key.interpolation, RawTrackInterpolation::KStep) as usize) << (i & 7)) as u8;
        }

        // Copy track's name.
        if !_input.name.is_empty() {
            track.name_ = _input.name.clone();
        }

        return Some(track);  // Success.
    }
}

//--------------------------------------------------------------------------------------------------
fn patch_begin_end_keys<ValueType: FloatType + FloatType<ImplType=ValueType>>(
    _input: &RawTrack<ValueType>, keyframes: &mut Vec<RawTrackKeyframe<ValueType>>) {
    if _input.keyframes.is_empty() {
        let default_value = TrackPolicy::<ValueType>::identity();

        let begin = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 0.0,
            value: default_value.clone(),
        };
        keyframes.push(begin);
        let end = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 1.0,
            value: default_value,
        };
        keyframes.push(end);
    } else if _input.keyframes.len() == 1 {
        let src_key = _input.keyframes.first().unwrap();
        let begin = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 0.0,
            value: src_key.value.clone(),
        };
        keyframes.push(begin);
        let end = RawTrackKeyframe::<ValueType> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 1.0,
            value: src_key.value.clone(),
        };
        keyframes.push(end);
    } else {
        // Copy all source data.
        // Push an initial and last keys if they don't exist.
        if _input.keyframes.first().unwrap().ratio != 0.0 {
            let src_key = _input.keyframes.first().unwrap();
            let begin = RawTrackKeyframe::<ValueType> {
                interpolation: RawTrackInterpolation::KLinear,
                ratio: 0.0,
                value: src_key.value.clone(),
            };
            keyframes.push(begin);
        }

        for i in 0.._input.keyframes.len() {
            keyframes.push(_input.keyframes[i].clone());
        }
        if _input.keyframes.last().unwrap().ratio != 1.0 {
            let src_key = _input.keyframes.last().unwrap();
            let end = RawTrackKeyframe::<ValueType> {
                interpolation: RawTrackInterpolation::KLinear,
                ratio: 1.0,
                value: src_key.value.clone(),
            };
            keyframes.push(end);
        }
    }
}

fn patch_begin_end_keys_quat(_input: &RawTrack<Quaternion>, keyframes: &mut Vec<RawTrackKeyframe<Quaternion>>) {
    if _input.keyframes.is_empty() {
        let default_value = TrackPolicy::<Quaternion>::identity();

        let begin = RawTrackKeyframe::<Quaternion> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 0.0,
            value: default_value.clone(),
        };
        keyframes.push(begin);
        let end = RawTrackKeyframe::<Quaternion> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 1.0,
            value: default_value,
        };
        keyframes.push(end);
    } else if _input.keyframes.len() == 1 {
        let src_key = _input.keyframes.first().unwrap();
        let begin = RawTrackKeyframe::<Quaternion> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 0.0,
            value: src_key.value.clone(),
        };
        keyframes.push(begin);
        let end = RawTrackKeyframe::<Quaternion> {
            interpolation: RawTrackInterpolation::KLinear,
            ratio: 1.0,
            value: src_key.value.clone(),
        };
        keyframes.push(end);
    } else {
        // Copy all source data.
        // Push an initial and last keys if they don't exist.
        if _input.keyframes.first().unwrap().ratio != 0.0 {
            let src_key = _input.keyframes.first().unwrap();
            let begin = RawTrackKeyframe::<Quaternion> {
                interpolation: RawTrackInterpolation::KLinear,
                ratio: 0.0,
                value: src_key.value.clone(),
            };
            keyframes.push(begin);
        }

        for i in 0.._input.keyframes.len() {
            keyframes.push(_input.keyframes[i].clone());
        }
        if _input.keyframes.last().unwrap().ratio != 1.0 {
            let src_key = _input.keyframes.last().unwrap();
            let end = RawTrackKeyframe::<Quaternion> {
                interpolation: RawTrackInterpolation::KLinear,
                ratio: 1.0,
                value: src_key.value.clone(),
            };
            keyframes.push(end);
        }
    }
}

// Fixes-up successive opposite quaternions that would fail to take the shortest
// path during the lerp.
fn fixup(_keyframes: &mut Vec<RawTrackKeyframe<Quaternion>>) {
    debug_assert!(_keyframes.len() >= 2);

    let identity = Quaternion::identity();
    for i in 0.._keyframes.len() {
        let prev_key: Quaternion;
        if i != 0 {
            prev_key = _keyframes[i - 1].value;
        } else {
            prev_key = Quaternion::new_default();
        }

        let src_key = &mut _keyframes[i].value;

        // Normalizes input quaternion.
        *src_key = src_key.normalize_safe(&identity);

        // Ensures quaternions are all on the same hemisphere.
        if i == 0 {
            if src_key.w < 0.0 {
                *src_key = -(src_key.clone());  // Q an -Q are the same rotation.
            }
        } else {
            let dot = src_key.x * prev_key.x + src_key.y * prev_key.y +
                src_key.z * prev_key.z + src_key.w * prev_key.w;
            if dot < 0.0 {
                *src_key = -(src_key.clone());  // Q an -Q are the same rotation.
            }
        }
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod track_builder {
    use crate::track_builder::TrackBuilder;
    use crate::track_sampling_job::*;
    use crate::raw_track::*;
    use crate::vec_float::*;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::quaternion::Quaternion;

    #[test]
    fn default() {
        {  // Building default RawFloatTrack succeeds.
            let raw_float_track = RawFloatTrack::new();
            assert_eq!(raw_float_track.validate(), true);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);
        }
    }

    #[test]
    fn build() {
        {  // Building a track with unsorted keys fails.
            let mut raw_float_track = RawFloatTrack::new();

            // Adds 2 unordered keys
            let first_key =
                RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                      0.8, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(first_key);
            let second_key =
                RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                      0.2, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(second_key);

            // Builds track
            assert_eq!(raw_float_track.validate(), false);
            assert_eq!(TrackBuilder::apply_float(&raw_float_track).is_none(), true);
        }

        {  // Building a track with invalid key frame's ratio fails.
            let mut raw_float_track = RawFloatTrack::new();

            // Adds 2 unordered keys
            let first_key =
                RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                      1.80, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            assert_eq!(raw_float_track.validate(), false);
            assert_eq!(TrackBuilder::apply_float(&raw_float_track).is_none(), true);
        }

        {  // Building a track with equal key frame's ratio fails.
            let mut raw_float_track = RawFloatTrack::new();

            // Adds 2 unordered keys
            let first_key =
                RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                      0.8, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(first_key);
            let second_key =
                RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                      0.80, Float::new_scalar(1.0));
            raw_float_track.keyframes.push(second_key);

            // Builds track
            assert_eq!(raw_float_track.validate(), false);
            assert_eq!(TrackBuilder::apply_float(&raw_float_track).is_none(), true);
        }

        {  // Building a valid track with 1 key succeeds.
            let mut raw_float_track = RawFloatTrack::new();
            let first_key =
                RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                      0.8, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(first_key);
            assert_eq!(raw_float_track.validate(), true);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);
        }
    }

    #[test]
    fn name() {
        {  // No name
            let raw_float_track = RawFloatTrack::new();

            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            assert_eq!(track.unwrap().name(), "");
        }

        {  // A name
            let mut raw_float_track = RawFloatTrack::new();
            raw_float_track.name = "test name".to_string();

            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            assert_eq!(track.unwrap().name().clone(), raw_float_track.name);
        }
    }

    #[test]
    fn build0keys() {
        let raw_float_track = RawFloatTrack::new();

        // Builds track
        let track = TrackBuilder::apply_float(&raw_float_track);
        assert_eq!(track.is_some(), true);

        // Samples to verify build output.
        let mut sampling = FloatTrackSamplingJob::new();
        sampling.track = track.as_ref();
        let mut result = Float::new_scalar(0.0);
        sampling.result = Some(&mut result);
        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
    }

    #[test]
    fn build_linear() {
        let mut result = Float::new_scalar(0.0);

        {  // 1 key at the beginning
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                                  0.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 1 key in the middle
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                                  0.5, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 1 key at the end
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                                  1.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 2 keys
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                                  0.5, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(RawTrackInterpolation::KLinear,
                                                   0.7, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            expect_near!(sampling.result.as_ref().unwrap().x, 23.0, 100.0*f32::EPSILON);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
        }

        {  // n keys with same value
            let mut raw_float_track = RawFloatTrack::new();

            let key1 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.5,
                                             Float::new_scalar(46.0));
            raw_float_track.keyframes.push(key1);
            let key2 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.7,
                                             Float::new_scalar(46.0));
            raw_float_track.keyframes.push(key2);
            let key3 = RawTrackKeyframe::new(RawTrackInterpolation::KLinear, 0.8,
                                             Float::new_scalar(46.0));
            raw_float_track.keyframes.push(key3);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.75;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }
    }

    #[test]
    fn build_step() {
        let mut result = Float::new_scalar(0.0);

        {  // 1 key at the beginning
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 1 key in the middle
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.5, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 1 key at the end
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  1.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 1 key at the beginning
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }

        {  // 2 keys
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                   0.7, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5 + 1.0e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7 - 1.0e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
        }

        {  // 3 keys
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.0, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                   0.7, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(second_key);
            let third_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  1.0, Float::new_scalar(99.0));
            raw_float_track.keyframes.push(third_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5 + 1.0e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7 - 1.0e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 0.9;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 99.0);
        }

        {  // 2 close keys
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.5, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KStep,
                0.5 + 2.0 * f32::EPSILON, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5 + f32::EPSILON;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5 + 2.0 * f32::EPSILON;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
        }

        {  // 3 keys
            let mut raw_float_track = RawFloatTrack::new();

            let first_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  0.5, Float::new_scalar(46.0));
            raw_float_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                   0.7, Float::new_scalar(0.0));
            raw_float_track.keyframes.push(second_key);
            let third_key = RawTrackKeyframe::new(RawTrackInterpolation::KStep,
                                                  1.0, Float::new_scalar(99.0));
            raw_float_track.keyframes.push(third_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_float_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.5 + 1e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7 - 1e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 0.7 + 1e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 1.0 - 1e-7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 99.0);
        }
    }

    #[test]
    fn build_mixed() {
        // Instantiates a builder objects with default parameters.
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

        sampling.ratio = 0.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 0.25;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 23.0);

        sampling.ratio = 0.5;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

        sampling.ratio = 0.5 + 1.0e-7;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

        sampling.ratio = 0.6;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

        sampling.ratio = 0.7 - 1.0e-7;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

        sampling.ratio = 0.7;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);

        sampling.ratio = 1.0;
        assert_eq!(sampling.run(), true);
        assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
    }

    #[test]
    fn float() {
        let mut raw_track = RawFloatTrack::new();

        {
            // Default value for quaternion is identity.
            let track = TrackBuilder::apply_float(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float::new_scalar(0.0);
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 0.0);
        }

        {
            let first_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.5, Float::new_scalar(23.0));
            raw_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.7, Float::new_scalar(46.0));
            raw_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float::new_scalar(0.0);
            let mut sampling = FloatTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 23.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 23.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            expect_near!(sampling.result.as_ref().unwrap().x, 34.5, 100.0*f32::EPSILON);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            assert_eq!(sampling.result.as_ref().unwrap().x, 46.0);
        }
    }

    #[test]
    fn float2() {
        let mut raw_track = RawFloat2Track::new();

        {
            // Default value for quaternion is identity.
            let track = TrackBuilder::apply_float2(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float2::new_default();
            let mut sampling = Float2TrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);

            expect_float2_eq!(result, 0.0, 0.0);
        }

        {
            let first_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.5, Float2::new(0.0, 23.0));
            raw_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.7, Float2::new(23.0, 46.0));
            raw_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float2(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float2::new_default();
            let mut sampling = Float2TrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            expect_float2_eq!(sampling.result.as_ref().unwrap(), 0.0, 23.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_float2_eq!(sampling.result.as_ref().unwrap(), 0.0, 23.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            expect_float2_eq!(sampling.result.as_ref().unwrap(), 11.5, 34.5);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            expect_float2_eq!(sampling.result.as_ref().unwrap(), 23.0, 46.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            expect_float2_eq!(sampling.result.as_ref().unwrap(), 23.0, 46.0);
        }
    }

    #[test]
    fn float3() {
        let mut raw_track = RawFloat3Track::new();

        {
            // Default value for quaternion is identity.
            let track = TrackBuilder::apply_float3(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float3::new_default();
            let mut sampling = Float3TrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_float3_eq!(result, 0.0, 0.0, 0.0);
        }

        {
            let first_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.5,
                Float3::new(0.0, 23.0, 46.0));
            raw_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.7,
                Float3::new(23.0, 46.0, 92.0));
            raw_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float3(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float3::new_default();
            let mut sampling = Float3TrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 23.0, 46.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_float3_eq!(sampling.result.as_ref().unwrap(), 0.0, 23.0, 46.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            expect_float3_eq!(sampling.result.as_ref().unwrap(), 11.5, 34.5, 69.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            expect_float3_eq!(sampling.result.as_ref().unwrap(), 23.0, 46.0, 92.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            expect_float3_eq!(sampling.result.as_ref().unwrap(), 23.0, 46.0, 92.0);
        }
    }

    #[test]
    fn float4() {
        let mut raw_track = RawFloat4Track::new();

        {
            // Default value for quaternion is identity.
            let track = TrackBuilder::apply_float4(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float4::new_default();
            let mut sampling = Float4TrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_float4_eq!(result, 0.0, 0.0, 0.0, 0.0);
        }

        {
            let first_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.5,
                Float4::new(0.0, 23.0, 46.0, 5.0));
            raw_track.keyframes.push(first_key);
            let second_key = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.7,
                Float4::new(23.0, 46.0, 92.0, 25.0));
            raw_track.keyframes.push(second_key);

            // Builds track
            let track = TrackBuilder::apply_float4(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Float4::new_default();
            let mut sampling = Float4TrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 23.0, 46.0, 5.0);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_float4_eq!(sampling.result.as_ref().unwrap(), 0.0, 23.0, 46.0, 5.0);

            sampling.ratio = 0.6;
            assert_eq!(sampling.run(), true);
            expect_float4_eq!(sampling.result.as_ref().unwrap(), 11.5, 34.5, 69.0, 15.0);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            expect_float4_eq!(sampling.result.as_ref().unwrap(), 23.0, 46.0, 92.0, 25.0);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            expect_float4_eq!(sampling.result.as_ref().unwrap(), 23.0, 46.0, 92.0, 25.0);
        }
    }

    #[test]
    fn quaternion() {
        let mut raw_track = RawQuaternionTrack::new();

        {
            // Default value for quaternion is identity.
            let track = TrackBuilder::apply_quaternion(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Quaternion::new_default();
            let mut sampling = QuaternionTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(result, 0.0, 0.0, 0.0, 1.0);
        }

        {
            let key0 = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.5,
                Quaternion::new(
                    -0.70710677, -0.0, -0.0,
                    -0.70710677));  // Will be opposed to be on the other hemisphere
            raw_track.keyframes.push(key0);
            let key1 = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.7,
                Quaternion::new(0.0, 0.70710677, 0.0, 0.70710677));
            raw_track.keyframes.push(key1);
            let key2 = RawTrackKeyframe::new(
                RawTrackInterpolation::KLinear, 0.8,
                Quaternion::new(-0.0, -0.70710677, -0.0, -0.70710677));
            raw_track.keyframes.push(key2);

            // Builds track
            let track = TrackBuilder::apply_quaternion(&raw_track);
            assert_eq!(track.is_some(), true);

            // Samples to verify build output.
            let mut result = Quaternion::new_default();
            let mut sampling = QuaternionTrackSamplingJob::new();
            sampling.track = track.as_ref();
            sampling.result = Some(&mut result);

            sampling.ratio = 0.0;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.70710677, 0.0, 0.0, 0.70710677);

            sampling.ratio = 0.5;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.70710677, 0.0, 0.0, 0.70710677);

            sampling.ratio = 0.54;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.61721331, 0.15430345, 0.0, 0.77151674);

            sampling.ratio = 0.7;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);

            sampling.ratio = 0.75;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);

            sampling.ratio = 0.8;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);

            sampling.ratio = 1.0;
            assert_eq!(sampling.run(), true);
            expect_quaternion_eq!(sampling.result.as_ref().unwrap(), 0.0, 0.70710677, 0.0, 0.70710677);
        }
    }
}