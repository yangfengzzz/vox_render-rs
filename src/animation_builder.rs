/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_animation::*;
use crate::animation::Animation;
use crate::vec_float::Float3;
use crate::quaternion::Quaternion;
use crate::animation_keyframe::Float3Key;
use std::cmp::Ordering;

// Defines the class responsible of building runtime animation instances from
// offline raw animations.
// No optimization at all is performed on the raw animation.
struct AnimationBuilder {}

impl AnimationBuilder {
    // Creates an Animation based on _raw_animation and *this builder parameters.
    // Returns a valid Animation on success.
    // See RawAnimation::Validate() for more details about failure reasons.
    // The animation is returned as an unique_ptr as ownership is given back to
    // the caller.
    pub fn apply(_raw_animation: &RawAnimation) -> Animation {
        todo!()
    }
}

//--------------------------------------------------------------------------------------------------
trait SortingType<T, Key: KeyType<T>> {
    fn new(track: u16, prev_key_time: f32, time: f32, value: T) -> Self;

    fn track(&self) -> u16;

    fn prev_key_time(&self) -> f32;

    fn key(&self) -> &Key;
}

struct SortingTranslationKey {
    track: u16,
    prev_key_time: f32,
    key: TranslationKey,
}

impl SortingType<Float3, TranslationKey> for SortingTranslationKey {
    fn new(track: u16, prev_key_time: f32, time: f32, value: Float3) -> Self {
        return SortingTranslationKey {
            track,
            prev_key_time,
            key: TranslationKey { time, value },
        };
    }

    fn track(&self) -> u16 {
        return self.track;
    }

    fn prev_key_time(&self) -> f32 {
        return self.prev_key_time;
    }

    fn key(&self) -> &TranslationKey {
        return &self.key;
    }
}

struct SortingRotationKey {
    track: u16,
    prev_key_time: f32,
    key: RotationKey,
}

impl SortingType<Quaternion, RotationKey> for SortingRotationKey {
    fn new(track: u16, prev_key_time: f32, time: f32, value: Quaternion) -> Self {
        return SortingRotationKey {
            track,
            prev_key_time,
            key: RotationKey { time, value },
        };
    }

    fn track(&self) -> u16 {
        return self.track;
    }

    fn prev_key_time(&self) -> f32 {
        return self.prev_key_time;
    }

    fn key(&self) -> &RotationKey {
        return &self.key;
    }
}

struct SortingScaleKey {
    track: u16,
    prev_key_time: f32,
    key: ScaleKey,
}

impl SortingType<Float3, ScaleKey> for SortingScaleKey {
    fn new(track: u16, prev_key_time: f32, time: f32, value: Float3) -> Self {
        return SortingScaleKey {
            track,
            prev_key_time,
            key: ScaleKey { time, value },
        };
    }

    fn track(&self) -> u16 {
        return self.track;
    }

    fn prev_key_time(&self) -> f32 {
        return self.prev_key_time;
    }

    fn key(&self) -> &ScaleKey {
        return &self.key;
    }
}

fn push_back_identity_key<T, _SrcKey: KeyType<T>, _DestKey: SortingType<T, _SrcKey>>(_track: u16, _time: f32, _dest: &mut Vec<_DestKey>) {
    let mut prev_time = -1.0;
    if !_dest.is_empty() && _dest.last().unwrap().track() == _track {
        prev_time = _dest.last().unwrap().key().time();
    }
    let key = _DestKey::new(_track, prev_time, _time, _SrcKey::identity());
    _dest.push(key);
}

// Copies a track from a RawAnimation to an Animation.
// Also fixes up the front (t = 0) and back keys (t = duration).
fn copy_raw<T, _SrcKey: KeyType<T>, _DestKey: SortingType<T, _SrcKey>>(_src: &Vec<_SrcKey>, _track: u16, _duration: f32,
                                                                       _dest: &mut Vec<_DestKey>) {
    if _src.len() == 0 {  // Adds 2 new keys.
        push_back_identity_key(_track, 0.0, _dest);
        push_back_identity_key(_track, _duration, _dest);
    } else if _src.len() == 1 {  // Adds 1 new key.
        let raw_key = _src.first().unwrap();
        debug_assert!(raw_key.time() >= 0.0 && raw_key.time() <= _duration);
        let first = _DestKey::new(_track, -1.0, 0.0, raw_key.value());
        _dest.push(first);
        let last = _DestKey::new(_track, 0.0, _duration, raw_key.value());
        _dest.push(last);
    } else {  // Copies all keys, and fixes up first and last keys.
        let mut prev_time = -1.0;
        if _src.first().unwrap().time() != 0.0 {  // Needs a key at t = 0.0.
            let first = _DestKey::new(_track, prev_time, 0.0, _src.first().unwrap().value());
            _dest.push(first);
            prev_time = 0.0;
        }
        for k in 0.._src.len() {  // Copies all keys.
            let raw_key = &_src[k];
            debug_assert!(raw_key.time() >= 0.0 && raw_key.time() <= _duration);
            let key = _DestKey::new(_track, prev_time, raw_key.time(), raw_key.value());
            _dest.push(key);
            prev_time = raw_key.time();
        }
        if _src.last().unwrap().time() - _duration != 0.0 {  // Needs a key at t = _duration.
            let last = _DestKey::new(_track, prev_time, _duration, _src.last().unwrap().value());
            _dest.push(last);
        }
    }
    debug_assert!(_dest.first().unwrap().key().time() == 0.0 && _dest.last().unwrap().key().time() - _duration == 0.0);
}


fn copy_to_animation<_SrcKey: KeyType<Float3>, _SortingKey: SortingType<Float3, _SrcKey>>(_src: &mut Vec<_SortingKey>,
                                                                                          _dest: &mut Vec<Float3Key>, _inv_duration: f32) {
    if _src.is_empty() {
        return;
    }

    // Sort animation keys to favor cache coherency.
    _src.sort_by(|_left, _right| {
        let time_diff = _left.prev_key_time() - _right.prev_key_time();
        return match time_diff < 0.0 || (time_diff == 0.0 && _left.track() < _right.track()) {
            true => Ordering::Less,
            false => Ordering::Greater,
        };
    });

    // Fills output.
    for i in 0.._src.len() {
        let key = &mut _dest[i];
        key.ratio = _src[i].key().time() * _inv_duration;
        key.track = _src[i].track();
        key.value[0] = crate::simd_math::float_to_half(_src[i].key().value().x);
        key.value[1] = crate::simd_math::float_to_half(_src[i].key().value().y);
        key.value[2] = crate::simd_math::float_to_half(_src[i].key().value().z);
    }
}










