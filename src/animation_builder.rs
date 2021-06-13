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
trait SortingType<Key: KeyType> {
    fn new(track: u16, prev_key_time: f32, time: f32, value: Key::T) -> Self;

    fn track(&self) -> u16;

    fn time(&self) -> f32;
}

struct SortingTranslationKey {
    track: u16,
    prev_key_time: f32,
    key: TranslationKey,
}

impl SortingType<TranslationKey> for SortingTranslationKey {
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

    fn time(&self) -> f32 {
        return self.key.time;
    }
}

struct SortingRotationKey {
    track: u16,
    prev_key_time: f32,
    key: RotationKey,
}

impl SortingType<RotationKey> for SortingRotationKey {
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

    fn time(&self) -> f32 {
        return self.key.time;
    }
}

struct SortingScaleKey {
    track: u16,
    prev_key_time: f32,
    key: ScaleKey,
}

impl SortingType<ScaleKey> for SortingScaleKey {
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

    fn time(&self) -> f32 {
        return self.key.time;
    }
}

fn push_back_identity_key<_SrcKey: KeyType, _DestKey: SortingType<_SrcKey>>(_track: u16, _time: f32, _dest: &mut Vec<_DestKey>) {
    let mut prev_time = -1.0;
    if !_dest.is_empty() && _dest.last().unwrap().track() == _track {
        prev_time = _dest.last().unwrap().time();
    }
    let key = _DestKey::new(_track, prev_time, _time, _SrcKey::identity());
    _dest.push(key);
}

// Copies a track from a RawAnimation to an Animation.
// Also fixes up the front (t = 0) and back keys (t = duration).
fn copy_raw<_SrcKey: KeyType, _DestKey: SortingType<_SrcKey>>(_src: &Vec<_SrcKey>, _track: u16, _duration: f32,
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
    debug_assert!(_dest.first().unwrap().time() == 0.0 && _dest.last().unwrap().time() - _duration == 0.0);
}












