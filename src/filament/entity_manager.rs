/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::filament::entity::Entity;
use std::collections::VecDeque;
use std::sync::Mutex;

trait EntityManagerListener {
    fn on_entities_destroyed(&self, entities: &[Entity]);
}

pub struct EntityManager {
    m_gens: [u8; EntityManager::RAW_INDEX_COUNT],

    m_current_index: u32,

    // stores indices that got freed
    m_free_list: Option<Mutex<VecDeque<u32>>>,
}

impl<'a> Default for EntityManager {
    fn default() -> Self {
        return EntityManager {
            m_gens: [0; EntityManager::RAW_INDEX_COUNT],
            m_current_index: 1,
            m_free_list: Default::default(),
        };
    }
}

impl EntityManager {
    // Get the global EntityManager. Is is recommended to cache this value.
    // Thread Safe.
    pub fn get() -> &'static mut EntityManager {
        // note: we leak the EntityManager because it's more important that it survives everything else
        // the leak is really not a problem because the process is terminating anyways.
        static mut INSTANCE: EntityManager = EntityManager {
            m_gens: [0; EntityManager::RAW_INDEX_COUNT],
            m_current_index: 1,
            m_free_list: None,
        };
        unsafe {
            if INSTANCE.m_free_list.is_none() {
                INSTANCE.m_free_list = Some(Default::default());
            }

            return &mut INSTANCE;
        }
    }

    // maximum number of entities that can exist at the same time
    pub fn get_max_entity_count() -> usize {
        // because index 0 is reserved, we only have 2^GENERATION_SHIFT - 1 valid indices
        return EntityManager::RAW_INDEX_COUNT - 1;
    }
}

pub const MIN_FREE_INDICES: usize = 1024;

impl EntityManager {
    // create n entities. Thread safe.
    fn create_impl(&mut self, entities: &mut [Entity]) {
        let mut index: u32;

        // this must be thread-safe, acquire the free-list mutex
        let lock = self.m_free_list.as_ref().unwrap().lock().unwrap();
        let mut current_index = self.m_current_index;
        for i in 0..entities.len() {
            if current_index >= EntityManager::RAW_INDEX_COUNT as u32 || lock.len() >= MIN_FREE_INDICES {
                if lock.is_empty() {
                    // return the null entity
                    entities[i] = Entity::default();
                    continue;
                }
                index = self.m_free_list.as_ref().unwrap().lock().unwrap().pop_front().unwrap()
            } else {
                // In the common case, we just grab the next index.
                // This works only until all indices have been used once, at which point
                // we're always in the slower case above. The idea is that we have enough indices
                // that it doesn't happen in practice.
                index = current_index;
                current_index += 1;
            }

            entities[i] = Entity::new(EntityManager::make_identity(self.m_gens[index as usize] as u32, index))
        }
        self.m_current_index = current_index;
    }

    // destroys n entities. Thread safe.
    fn destroy_impl(&mut self, entities: &mut [Entity]) {
        for i in 0..entities.len() {
            // ... deleting a dead Entity will corrupt the internal state, so we protect ourselves
            // against it. We don't guarantee anything about external state -- e.g. the listeners
            // will be called.
            if self.is_alive(entities[i]) {
                let index = EntityManager::get_index(entities[i]);
                self.m_free_list.as_mut().unwrap().lock().unwrap().push_back(index);

                // The generation update doesn't require the lock because it's only used for isAlive()
                // and entities work as weak references -- it just means that isAlive() could return
                // true a little longer than expected in some other threads.
                // We do need a memory fence though, it is provided by the mFreeListLock.unlock() below.
                self.m_gens[index as usize] += 1;
            }
        }
    }
}

impl EntityManager {
    // create a new Entity. Thread safe.
    // Return Entity.isNull() if the entity cannot be allocated.
    pub fn create(&mut self) -> Entity {
        let e = Entity::default();
        self.create_impl(&mut [e]);
        return e;
    }

    // destroys an Entity. Thread safe.
    pub fn destroy(&mut self, e: Entity) {
        self.destroy_impl(&mut [e]);
    }

    // return whether the given Entity has been destroyed (false) or not (true).
    // Thread safe.
    pub fn is_alive(&self, e: Entity) -> bool {
        debug_assert!(EntityManager::get_index(e) < EntityManager::RAW_INDEX_COUNT as u32);
        return (!e.is_null()) && (EntityManager::get_generation(e) == self.m_gens[EntityManager::get_index(e) as usize] as u32);
    }

    // current generation of the given index. Use for debugging and testing.
    pub fn get_generation_for_index(&self, index: usize) -> u8 {
        return self.m_gens[index];
    }
}

impl EntityManager {
    // GENERATION_SHIFT determines how many simultaneous Entities are available, the
    // minimum memory requirement is 2^GENERATION_SHIFT bytes.
    const GENERATION_SHIFT: i32 = 17;
    const RAW_INDEX_COUNT: usize = (1 << EntityManager::GENERATION_SHIFT);
    const INDEX_MASK: u32 = (1 << EntityManager::GENERATION_SHIFT) - 1;

    pub fn get_generation(e: Entity) -> u32 {
        return e.get_id() >> EntityManager::GENERATION_SHIFT;
    }
    pub fn get_index(e: Entity) -> u32 {
        return e.get_id() & EntityManager::INDEX_MASK;
    }
    pub fn make_identity(g: u32, i: u32) -> u32 {
        return (g << EntityManager::GENERATION_SHIFT) | (i & EntityManager::INDEX_MASK);
    }
}

#[cfg(test)]
mod test {
    use crate::filament::entity_manager::EntityManager;

    #[test]
    fn create() {
        let entity = EntityManager::get().create();
    }
}