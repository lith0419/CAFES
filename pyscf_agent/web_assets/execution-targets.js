(function exposeExecutionTargetSelector(global) {
  'use strict';

  const STORAGE_KEY = 'pyscf-agent-execution-target';
  const RESOURCE_STORAGE_PREFIX = 'pyscf-agent-resource-profile:';
  let defaultTarget = 'local';
  let availableTargets = [];
  let selectedProfile = 'auto';
  let profileLoadSerial = 0;

  function storedTarget() {
    try {
      return global.localStorage.getItem(STORAGE_KEY) || '';
    } catch (error) {
      return '';
    }
  }

  function rememberTarget(targetId) {
    try {
      global.localStorage.setItem(STORAGE_KEY, targetId);
    } catch (error) {
      // The selector remains usable when browser storage is unavailable.
    }
  }

  function targetSelect(selectId) {
    return document.getElementById(selectId || 'execution-target');
  }

  function resourceSelect(selectId) {
    return document.getElementById(selectId || 'resource-profile');
  }

  function renderRuntimeBinding(targetId) {
    const detail = document.getElementById('runtime-binding-detail');
    if (!detail) return;
    const target = availableTargets.find((item) => String(item.id) === String(targetId));
    if (!target || !target.runtime_match_required) {
      detail.textContent = '';
      return;
    }
    const releaseId = String(target.runtime_release_id || 'missing');
    detail.textContent = `Bound release: ${releaseId}`;
  }

  function storedProfile(targetId) {
    try {
      return global.localStorage.getItem(`${RESOURCE_STORAGE_PREFIX}${targetId}`) || '';
    } catch (error) {
      return '';
    }
  }

  function rememberProfile(targetId, profileId) {
    try {
      global.localStorage.setItem(`${RESOURCE_STORAGE_PREFIX}${targetId}`, profileId);
    } catch (error) {
      // The selector remains usable when browser storage is unavailable.
    }
  }

  function renderTargets(select, payload) {
    availableTargets = Array.isArray(payload.targets) ? payload.targets : [];
    defaultTarget = String(payload.default_target || 'local');
    if (!availableTargets.length) {
      availableTargets = [{ id: 'local', label: 'Local', location: 'current_process' }];
      defaultTarget = 'local';
    }

    const previous = storedTarget();
    const targetIds = new Set(availableTargets.map((target) => String(target.id)));
    const selected = targetIds.has(previous)
      ? previous
      : (targetIds.has(defaultTarget) ? defaultTarget : String(availableTargets[0].id));

    select.replaceChildren();
    availableTargets.forEach((target) => {
      const option = document.createElement('option');
      option.value = String(target.id);
      option.textContent = String(target.label || target.id);
      option.dataset.location = String(target.location || '');
      select.appendChild(option);
    });
    select.value = selected;
    select.disabled = false;
    rememberTarget(selected);
    renderRuntimeBinding(selected);
  }

  function renderProfiles(select, payload, targetId) {
    const profiles = Array.isArray(payload.profiles) && payload.profiles.length
      ? payload.profiles
      : [{ id: 'auto', label: 'Auto' }];
    const ids = new Set(profiles.map((profile) => String(profile.id)));
    const previous = storedProfile(targetId);
    const defaultProfile = String(payload.default_profile || 'auto');
    selectedProfile = ids.has(previous)
      ? previous
      : (ids.has(defaultProfile) ? defaultProfile : String(profiles[0].id));
    select.replaceChildren();
    profiles.forEach((profile) => {
      const option = document.createElement('option');
      option.value = String(profile.id);
      option.textContent = String(profile.label || profile.id);
      select.appendChild(option);
    });
    select.value = selectedProfile;
    select.disabled = profiles.length <= 1;
    select.title = '';
    rememberProfile(targetId, selectedProfile);
  }

  async function initializeResourceProfileSelect(targetId, selectId) {
    const select = resourceSelect(selectId);
    if (!select) {
      return null;
    }
    const normalizedTarget = String(targetId || selectedExecutionTarget());
    const loadSerial = ++profileLoadSerial;
    selectedProfile = 'auto';
    select.disabled = true;
    select.replaceChildren(new Option('Loading...', ''));
    try {
      const query = new URLSearchParams({ execution_target: normalizedTarget });
      const response = await fetch(`/api/resource-profiles?${query.toString()}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Resource profiles are unavailable');
      }
      if (loadSerial !== profileLoadSerial || normalizedTarget !== selectedExecutionTarget()) {
        return null;
      }
      renderProfiles(select, payload, normalizedTarget);
    } catch (error) {
      if (loadSerial !== profileLoadSerial || normalizedTarget !== selectedExecutionTarget()) {
        return null;
      }
      renderProfiles(select, {
        default_profile: 'auto',
        profiles: [{ id: 'auto', label: 'Auto' }],
      }, normalizedTarget);
      select.title = error instanceof Error ? error.message : String(error);
    }
    if (select.dataset.profileListener !== 'true') {
      select.addEventListener('change', () => {
        selectedProfile = select.value || 'auto';
        rememberProfile(selectedExecutionTarget(), selectedProfile);
      });
      select.dataset.profileListener = 'true';
    }
    return select.value;
  }

  async function initializeExecutionTargetSelect(selectId) {
    const select = targetSelect(selectId);
    if (!select) {
      return null;
    }
    select.disabled = true;
    select.replaceChildren(new Option('Loading...', ''));
    try {
      const response = await fetch('/api/execution-targets');
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || 'Execution targets are unavailable');
      }
      renderTargets(select, payload);
    } catch (error) {
      renderTargets(select, {
        default_target: 'local',
        targets: [{ id: 'local', label: 'Local', location: 'current_process' }],
      });
      select.title = error instanceof Error ? error.message : String(error);
    }
    if (select.dataset.targetListener !== 'true') {
      select.addEventListener('change', () => {
        rememberTarget(select.value);
        renderRuntimeBinding(select.value);
        initializeResourceProfileSelect(select.value);
      });
      select.dataset.targetListener = 'true';
    }
    await initializeResourceProfileSelect(select.value);
    return select.value;
  }

  function selectedExecutionTarget(selectId) {
    const select = targetSelect(selectId);
    return select && select.value ? select.value : defaultTarget;
  }

  function selectedResourceProfile(selectId) {
    const select = resourceSelect(selectId);
    const value = select && select.value ? select.value : selectedProfile;
    return value && value !== 'auto' ? value : null;
  }

  global.initializeExecutionTargetSelect = initializeExecutionTargetSelect;
  global.initializeResourceProfileSelect = initializeResourceProfileSelect;
  global.selectedExecutionTarget = selectedExecutionTarget;
  global.selectedResourceProfile = selectedResourceProfile;
})(window);
