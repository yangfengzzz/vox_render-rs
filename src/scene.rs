/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use cgmath::Vector2;
use crate::entity::Entity;
use std::sync::{RwLock, Arc};
use crate::engine::Engine;

pub struct Scene {
    /** scene name */
    pub name: String,

    pub _is_active_in_engine: bool,

    _destroyed: bool,
    _root_entities: Vec<Arc<RwLock<Entity>>>,
    _resolution: Vector2<f32>,

    _engine: Arc<RwLock<Engine>>,
}

impl Scene {}

impl Scene {
    /**
     * Count of root entities.
     * @readonly
     */
    pub fn root_entities_count(&self) -> usize {
        return self._root_entities.len();
    }

    /**
     * Root entity collection.
     * @readonly
     */
    pub fn root_entities(&self) -> &Vec<Arc<RwLock<Entity>>> {
        return &self._root_entities;
    }

    /**
     * Whether it's destroyed.
     * @readonly
     */
    pub fn destroyed(&self) -> bool {
        return self._destroyed;
    }
}

impl Scene {
    /**
     * Create root entity.
     * @param name - Entity name
     * @returns Entity
     */
    pub fn create_root_entity(&self, name: Option<String>) -> Arc<RwLock<Entity>> {
        let entity = Arc::new(RwLock::new(Entity::new(self._engine.clone(), name)));
        self.addRootEntity(entity);
        return entity.clone();
    }

    /**
     * Append an entity.
     * @param entity - The root entity to add
     */
    pub fn add_root_entity(entity: Arc<RwLock<Entity>>) {
        let is_root = entity.read().unwrap()._is_root;

        // let entity become root
        if !is_root {
            entity.write().unwrap()._is_root = true;
            entity._removeFromParent();
        }
    }
}