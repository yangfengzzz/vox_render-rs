/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_animation::*;
use crate::animation::Animation;
use crate::vec_float::{Float3, Float4};
use crate::quaternion::Quaternion;
use crate::animation_keyframe::{Float3Key, QuaternionKey};
use std::cmp::Ordering;

// Defines the class responsible of building runtime animation instances from
// offline raw animations.
// No optimization at all is performed on the raw animation.
pub struct AnimationBuilder {}

impl AnimationBuilder {
    // Creates an Animation based on _raw_animation and *this builder parameters.
    // Returns a valid Animation on success.
    // See RawAnimation::validate() for more details about failure reasons.
    // The animation is returned as an unique_ptr as ownership is given back to
    // the caller.
    pub fn apply(_input: &RawAnimation) -> Option<Animation> {
        // Tests _raw_animation validity.
        if !_input.validate() {
            return None;
        }

        // Everything is fine, allocates and fills the animation.
        // Nothing can fail now.
        let mut animation = Animation::new();

        // Sets duration.
        let duration = _input.duration;
        let inv_duration = 1.0 / _input.duration;
        animation.duration_ = duration;
        // A _duration == 0 would create some division by 0 during sampling.
        // Also we need at least to keys with different times, which cannot be done
        // if duration is 0.
        debug_assert!(duration > 0.0);  // This case is handled by validate().

        // Sets tracks count. Can be safely casted to uint16_t as number of tracks as
        // already been validated.
        let num_tracks = _input.num_tracks() as u16;
        animation.num_tracks_ = num_tracks as i32;
        let num_soa_tracks = align(num_tracks, 4);

        // Declares and preallocates tracks to sort.
        let mut translations: usize = 0;
        let mut rotations: usize = 0;
        let mut scales: usize = 0;
        for i in 0..num_tracks as usize {
            let raw_track = &_input.tracks[i];
            translations += raw_track.translations.len() + 2;  // +2 because worst case
            rotations += raw_track.rotations.len() + 2;        // needs to add the
            scales += raw_track.scales.len() + 2;              // first and last keys.
        }
        let mut sorting_translations: Vec<SortingTranslationKey> = Vec::new();
        sorting_translations.reserve(translations);
        let mut sorting_rotations: Vec<SortingRotationKey> = Vec::new();
        sorting_rotations.reserve(rotations);
        let mut sorting_scales: Vec<SortingScaleKey> = Vec::new();
        sorting_scales.reserve(scales);

        // Filters RawAnimation keys and copies them to the output sorting structure.
        for i in 0..num_tracks {
            let raw_track = &_input.tracks[i as usize];
            copy_raw(&raw_track.translations, i, duration, &mut sorting_translations);
            copy_raw(&raw_track.rotations, i, duration, &mut sorting_rotations);
            copy_raw(&raw_track.scales, i, duration, &mut sorting_scales);
        }

        // Add enough identity keys to match soa requirements.
        for i in 0..num_soa_tracks {
            push_back_identity_key(i, 0.0, &mut sorting_translations);
            push_back_identity_key(i, duration, &mut sorting_translations);

            push_back_identity_key(i, 0.0, &mut sorting_rotations);
            push_back_identity_key(i, duration, &mut sorting_rotations);

            push_back_identity_key(i, 0.0, &mut sorting_scales);
            push_back_identity_key(i, duration, &mut sorting_scales);
        }

        // Allocate animation members.
        animation.allocate(sorting_translations.len(),
                           sorting_rotations.len(),
                           sorting_scales.len());

        // Copy sorted keys to final animation.
        copy_to_animation(&mut sorting_translations, &mut animation.translations_,
                          inv_duration);
        copy_to_animation_quat(&mut sorting_rotations, &mut animation.rotations_,
                               inv_duration);
        copy_to_animation(&mut sorting_scales, &mut animation.scales_,
                          inv_duration);

        // Copy animation's name.
        if animation.name_.is_none() {
            animation.name_ = Some(_input.name.clone());
        }

        return Some(animation);  // Success.
    }
}

pub fn align(_value: u16, _alignment: usize) -> u16 {
    return ((_value as usize + (_alignment - 1)) & _alignment.overflowing_neg().0) as u16;
}

//--------------------------------------------------------------------------------------------------
trait SortingType<T, Key: KeyType<T>> {
    fn new(track: u16, prev_key_time: f32, time: f32, value: T) -> Self;

    fn track(&self) -> u16;

    fn prev_key_time(&self) -> f32;

    fn key(&self) -> &Key;

    fn key_mut(&mut self) -> &mut Key;
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

    fn key_mut(&mut self) -> &mut TranslationKey {
        return &mut self.key;
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

    fn key_mut(&mut self) -> &mut RotationKey {
        return &mut self.key;
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

    fn key_mut(&mut self) -> &mut ScaleKey {
        return &mut self.key;
    }
}

fn push_back_identity_key<T, _SrcKey: KeyType<T>, _DestKey: SortingType<T, _SrcKey>>(
    _track: u16, _time: f32, _dest: &mut Vec<_DestKey>) {
    let mut prev_time = -1.0;
    if !_dest.is_empty() && _dest.last().unwrap().track() == _track {
        prev_time = _dest.last().unwrap().key().time();
    }
    let key = _DestKey::new(_track, prev_time, _time, _SrcKey::identity());
    _dest.push(key);
}

// Copies a track from a RawAnimation to an Animation.
// Also fixes up the front (t = 0) and back keys (t = duration).
fn copy_raw<T, _SrcKey: KeyType<T>, _DestKey: SortingType<T, _SrcKey>>(
    _src: &Vec<_SrcKey>, _track: u16, _duration: f32, _dest: &mut Vec<_DestKey>) {
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


fn copy_to_animation<_SrcKey: KeyType<Float3>, _SortingKey: SortingType<Float3, _SrcKey>>(
    _src: &mut Vec<_SortingKey>, _dest: &mut Vec<Float3Key>, _inv_duration: f32) {
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

// Compresses quaternion to ozz::animation::RotationKey format.
// The 3 smallest components of the quaternion are quantized to 16 bits
// integers, while the largest is recomputed thanks to quaternion normalization
// property (x^2+y^2+z^2+w^2 = 1). Because the 3 components are the 3 smallest,
// their value cannot be greater than sqrt(2)/2. Thus quantization quality is
// improved by pre-multiplying each component by sqrt(2).
fn compress_quat(_src: &Quaternion, _dest: &mut QuaternionKey) {
    // Finds the largest quaternion component.
    let quat = [_src.x, _src.y, _src.z, _src.w];
    let largest = quat.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index).unwrap();
    debug_assert!(largest <= 3);
    _dest.largest = (largest & 0x3) as u16;

    // Stores the sign of the largest component.
    _dest.sign = u16::from(quat[largest] < 0.0);

    // Quantize the 3 smallest components on 16 bits signed integers.
    let k_float2int = 32767.0 * crate::math_constant::K_SQRT2;
    let k_mapping = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    let map = k_mapping[largest];
    let a = f32::floor(quat[map[0]] * k_float2int + 0.5) as i32;
    let b = f32::floor(quat[map[1]] * k_float2int + 0.5) as i32;
    let c = f32::floor(quat[map[2]] * k_float2int + 0.5) as i32;
    _dest.value[0] = (i32::clamp(-32767, a, 32767) & 0xffff) as i16;
    _dest.value[1] = (i32::clamp(-32767, b, 32767) & 0xffff) as i16;
    _dest.value[2] = (i32::clamp(-32767, c, 32767) & 0xffff) as i16;
}

fn copy_to_animation_quat<_SrcKey: KeyType<Quaternion>, _SortingKey: SortingType<Quaternion, _SrcKey>>(
    _src: &mut Vec<_SortingKey>, _dest: &mut Vec<QuaternionKey>, _inv_duration: f32) {
    if _src.is_empty() {
        return;
    }

    // Normalize quaternions.
    // Also fixes-up successive opposite quaternions that would fail to take the
    // shortest path during the normalized-lerp.
    // Note that keys are still sorted per-track at that point, which allows this
    // algorithm to process all consecutive keys.
    let mut track = usize::MAX;
    let identity = Quaternion::identity();
    for i in 0.._src.len() {
        let mut normalized = _src[i].key().value().normalize_safe(&identity);
        if track != _src[i].track() as usize {   // First key of the track.
            if normalized.w < 0.0 {    // .w eq to a dot with identity quaternion.
                normalized = -normalized;  // Q an -Q are the same rotation.
            }
        } else {  // Still on the same track: so fixes-up quaternion.
            let prev = Float4::new(_src[i - 1].key().value().x, _src[i - 1].key().value().y,
                                   _src[i - 1].key().value().z, _src[i - 1].key().value().w);
            let curr = Float4::new(normalized.x, normalized.y, normalized.z,
                                   normalized.w);
            if prev.dot(&curr) < 0.0 {
                normalized = -normalized;  // Q an -Q are the same rotation.
            }
        }
        // Stores fixed-up quaternion.
        _src[i].key_mut().set_value(normalized);
        track = _src[i].track() as usize;

        //Sort
        _src.sort_by(|_left, _right| {
            let time_diff = _left.prev_key_time() - _right.prev_key_time();
            return match time_diff < 0.0 || (time_diff == 0.0 && _left.track() < _right.track()) {
                true => Ordering::Less,
                false => Ordering::Greater,
            };
        });

        // Fills rotation keys output.
        for i in 0.._src.len() {
            let skey = &_src[i];
            let dkey = &mut _dest[i];
            dkey.ratio = skey.key().time() * _inv_duration;
            dkey.track = skey.track();

            // Compress quaternion to destination container.
            compress_quat(&skey.key().value(), dkey);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod animation_builder {
    use crate::raw_animation::*;
    use crate::animation_builder::AnimationBuilder;
    use crate::vec_float::Float3;
    use crate::math_test_helper::*;
    use crate::simd_math::*;
    use crate::*;
    use crate::soa_transform::SoaTransform;

    #[test]
    fn error() {
        {  // Building an empty Animation fails because animation duration
            // must be >= 0.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = -1.0;  // Negative duration.
            assert_eq!(raw_animation.validate(), false);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_none(), true);
        }

        {  // Building an empty Animation fails because animation duration
            // must be >= 0.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 0.0;  // Invalid duration.
            assert_eq!(raw_animation.validate(), false);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_none(), true);
        }

        {  // Building an animation with too much tracks fails.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(crate::skeleton::Constants::KMaxJoints as usize + 1, JointTrack::new());
            assert_eq!(raw_animation.validate(), false);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_none(), true);
        }

        {  // Building default animation succeeds.
            let raw_animation = RawAnimation::new();
            assert_eq!(raw_animation.duration, 1.0);
            assert_eq!(raw_animation.validate(), true);

            // Builds animation
            let anim = AnimationBuilder::apply(&raw_animation);
            assert_eq!(anim.is_some(), true);
        }

        {  // Building an animation with max joints succeeds.
            let mut raw_animation = RawAnimation::new();
            raw_animation.tracks.resize(crate::skeleton::Constants::KMaxJoints as usize, JointTrack::new());
            assert_eq!(raw_animation.num_tracks(), crate::skeleton::Constants::KMaxJoints as i32);
            assert_eq!(raw_animation.validate(), true);

            // Builds animation
            let anim = AnimationBuilder::apply(&raw_animation);
            assert_eq!(anim.is_some(), true);
        }
    }

    #[test]
    fn build() {
        {  // Building an Animation with unsorted keys fails.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(1, JointTrack::new());

            // Adds 2 unordered keys
            let first_key = TranslationKey { time: 0.8, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(first_key);
            let second_key = TranslationKey { time: 0.2, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(second_key);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_some(), false);
        }

        {  // Building an Animation with invalid key frame's time fails.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(1, JointTrack::new());

            // Adds a key with a time greater than animation duration.
            let first_key = TranslationKey { time: 2.0, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(first_key);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_some(), false);
        }

        {  // Building an Animation with unsorted key frame's time fails.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(2, JointTrack::new());

            // Adds 2 unsorted keys.
            let first_key = TranslationKey { time: 0.7, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(first_key);
            let second_key = TranslationKey { time: 0.1, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(second_key);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_some(), false);
        }

        {  // Building an Animation with equal key frame's time fails.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(2, JointTrack::new());

            // Adds 2 unsorted keys.
            let key = TranslationKey { time: 0.7, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(key.clone());
            raw_animation.tracks[0].translations.push(key);

            // Builds animation
            assert_eq!(AnimationBuilder::apply(&raw_animation).is_some(), false);
        }

        {  // Building a valid Animation with empty tracks succeeds.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 46.0;
            raw_animation.tracks.resize(46, JointTrack::new());

            // Builds animation
            let anim = AnimationBuilder::apply(&raw_animation);
            assert_eq!(anim.is_some(), true);
            assert_eq!(anim.as_ref().unwrap().duration(), 46.0);
            assert_eq!(anim.as_ref().unwrap().num_tracks(), 46);
        }

        {  // Building a valid Animation with 1 track succeeds.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 46.0;
            raw_animation.tracks.resize(1, JointTrack::new());

            let first_key = TranslationKey { time: 0.7, value: Float3::zero() };
            raw_animation.tracks[0].translations.push(first_key);

            // Builds animation
            let anim = AnimationBuilder::apply(&raw_animation);
            assert_eq!(anim.is_some(), true);
            assert_eq!(anim.as_ref().unwrap().duration(), 46.0);
            assert_eq!(anim.as_ref().unwrap().num_tracks(), 1);
        }
    }

    #[test]
    fn name() {
        {  // Building an unnamed animation.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(46, JointTrack::new());

            // Builds animation
            let anim = AnimationBuilder::apply(&raw_animation);
            assert_eq!(anim.is_some(), true);

            // Should
            assert_eq!(anim.as_ref().unwrap().name(), "");
        }

        {  // Building an unnamed animation.
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(46, JointTrack::new());
            raw_animation.name = "46".to_string();

            // Builds animation
            let anim = AnimationBuilder::apply(&raw_animation);
            assert_eq!(anim.is_some(), true);

            // Should
            assert_eq!(anim.as_ref().unwrap().name(), "46");
        }
    }

    #[test]
    fn sort() {
        {
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(4, JointTrack::new());

            // Raw animation inputs.
            //     0              1
            // --------------------
            // 0 - A     B        |
            // 1 - C  D  E        |
            // 2 - F  G     H  I  J
            // 3 - K  L  M  N     |

            // Final animation.
            //     0              1
            // --------------------
            // 0 - 0     4       11
            // 1 - 1  5  8       12
            // 2 - 2  6     9 14 16
            // 3 - 3  7 10 13    15

            let a = TranslationKey {
                time: 0.0 * raw_animation.duration,
                value: Float3::new(1.0, 0.0, 0.0),
            };
            raw_animation.tracks[0].translations.push(a);

            let b = TranslationKey {
                time: 0.4 * raw_animation.duration,
                value: Float3::new(3.0, 0.0, 0.0),
            };
            raw_animation.tracks[0].translations.push(b);

            let c = TranslationKey {
                time: 0.0 * raw_animation.duration,
                value: Float3::new(2.0, 0.0, 0.0),
            };
            raw_animation.tracks[1].translations.push(c);

            let d = TranslationKey {
                time: 0.2 * raw_animation.duration,
                value: Float3::new(6.0, 0.0, 0.0),
            };
            raw_animation.tracks[1].translations.push(d);

            let e = TranslationKey {
                time: 0.4 * raw_animation.duration,
                value: Float3::new(8.0, 0.0, 0.0),
            };
            raw_animation.tracks[1].translations.push(e);

            let f = TranslationKey {
                time: 0.0 * raw_animation.duration,
                value: Float3::new(12.0, 0.0, 0.0),
            };
            raw_animation.tracks[2].translations.push(f);

            let g = TranslationKey {
                time: 0.2 * raw_animation.duration,
                value: Float3::new(11.0, 0.0, 0.0),
            };
            raw_animation.tracks[2].translations.push(g);

            let h = TranslationKey {
                time: 0.6 * raw_animation.duration,
                value: Float3::new(9.0, 0.0, 0.0),
            };
            raw_animation.tracks[2].translations.push(h);

            let i = TranslationKey {
                time: 0.8 * raw_animation.duration,
                value: Float3::new(7.0, 0.0, 0.0),
            };
            raw_animation.tracks[2].translations.push(i);

            let j = TranslationKey {
                time: 1.0 * raw_animation.duration,
                value: Float3::new(5.0, 0.0, 0.0),
            };
            raw_animation.tracks[2].translations.push(j);

            let k = TranslationKey {
                time: 0.0 * raw_animation.duration,
                value: Float3::new(1.0, 0.0, 0.0),
            };
            raw_animation.tracks[3].translations.push(k);

            let l = TranslationKey {
                time: 0.2 * raw_animation.duration,
                value: Float3::new(2.0, 0.0, 0.0),
            };
            raw_animation.tracks[3].translations.push(l);

            let m = TranslationKey {
                time: 0.4 * raw_animation.duration,
                value: Float3::new(3.0, 0.0, 0.0),
            };
            raw_animation.tracks[3].translations.push(m);

            let n = TranslationKey {
                time: 0.6 * raw_animation.duration,
                value: Float3::new(4.0, 0.0, 0.0),
            };
            raw_animation.tracks[3].translations.push(n);

            // Builds animation
            let animation = AnimationBuilder::apply(&raw_animation);
            assert_eq!(animation.is_some(), true);

            // Duration must be maintained.
            assert_eq!(animation.as_ref().unwrap().duration(), raw_animation.duration);

            // Needs to sample to test the animation.
            let mut job = crate::sampling_job::SamplingJob::new();
            job.cache.resize(1);
            job.output.resize(1, SoaTransform::identity());
            job.animation = animation.as_ref();
            // Samples and compares the two animations
            {  // Samples at t = 0
                job.ratio = 0.0;
                job.run();

                println!("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                         job.output[0].translation.x.get_x(),
                         job.output[0].translation.x.get_y(),
                         job.output[0].translation.x.get_z(),
                         job.output[0].translation.x.get_w(),
                         job.output[0].translation.y.get_x(),
                         job.output[0].translation.y.get_y(),
                         job.output[0].translation.y.get_z(),
                         job.output[0].translation.y.get_w(),
                         job.output[0].translation.z.get_x(),
                         job.output[0].translation.z.get_y(),
                         job.output[0].translation.z.get_z(),
                         job.output[0].translation.z.get_w());

                expect_soa_float3_eq_est!(job.output[0].translation, 1.0, 2.0, 12.0, 1.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
            {  // Samples at t = .2
                job.ratio = 0.2;
                job.run();
                expect_soa_float3_eq_est!(job.output[0].translation, 2.0, 6.0, 11.0, 2.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
            {  // Samples at t = .4
                job.ratio = 0.4;
                job.run();
                expect_soa_float3_eq_est!(job.output[0].translation, 3.0, 8.0, 10.0, 3.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
            {  // Samples at t = .6
                job.ratio = 0.6;
                job.run();
                expect_soa_float3_eq_est!(job.output[0].translation, 3.0, 8.0, 9.0, 4.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
            {  // Samples at t = .8
                job.ratio = 0.8;
                job.run();
                expect_soa_float3_eq_est!(job.output[0].translation, 3.0, 8.0, 7.0, 4.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
            {  // Samples at t = 1
                job.ratio = 1.0;
                job.run();
                expect_soa_float3_eq_est!(job.output[0].translation, 3.0, 8.0, 5.0, 4.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
        }
    }
}