/**
 * Story & Roleplay Workbench - Vue 3 Application Logic
 *
 * Provides reactive workbench state and CRUD operations for:
 * - Character Cards management (V2 / Tavern spec compatible)
 * - Lorebooks and keyword-triggered world context entries
 * - Roleplay and Story workbench defaults & event stubs
 */

const AppDefinition = {
  data() {
    return {
      // Global navigation & application health
      activeTab: 'roleplay', // 'roleplay' | 'group' | 'story' | 'characters' | 'groups' | 'lorebooks'
      backendOnline: false,
      healthInterval: null,
      toast: {
        show: false,
        message: '',
        type: 'info', // 'info' | 'success' | 'error'
      },
      toastTimer: null,

      // =======================================================================
      // Characters Management State
      // =======================================================================
      characters: [], // Array of CharacterCard objects
      selectedCharId: null,
      charSearchQuery: '',
      charSearch: '', // Alias
      charTagsInput: '',
      charForm: {
        char_id: '',
        name: '',
        tags_str: '',
        tags: [],
        description: '',
        personality: '',
        scenario: '',
        first_mes: '',
        alternate_greetings: [],
        mes_example: '',
        system_prompt: '',
        post_history_instructions: '',
        creator_notes: '',
        lorebook_id: '',
      },

      // =======================================================================
      // Lorebooks Management State
      // =======================================================================
      lorebooks: {}, // Object mapping lorebook_id -> lorebook object
      selectedLorebookId: null,
      lorebookSearchQuery: '',
      lbSearch: '', // Alias
      lorebookForm: {
        name: '',
        description: '',
        entries: [],
      },

      // =======================================================================
      // Groups Management State
      // =======================================================================
      groups: {}, // Object mapping group_id -> GroupCard
      selectedGroupId: null,
      groupSearchQuery: '',
      groupForm: { group_id: '', name: '', char_ids: [], scenario: '', first_mes: '', lorebook_id: '' },

      // =======================================================================
      // Roleplay Tab Reactive State (Task 6)
      // =======================================================================
      rpSessionId: 'sess_' + Math.random().toString(36).substring(2, 10),
      rpCharId: '',
      rpUserName: 'User',
      rpAuthorsNote: '',
      rpLorebookId: '',
      rpLorebookSentKey: null, // "<session>|<lorebook>" last loaded into the backend session
      rpChunkSize: 1,
      rpMessages: [],
      selectedGreetingIndex: 0,
      isGeneratingRP: false,
      rpAbortController: null,
      rpInput: '',
      rpSessions: [], // past chats with the selected character, newest first
      showRPHistory: false,
      // History title being renamed (RP and story lists share it; only one row edits at a time).
      editingSessionId: null,
      editingTitle: '',

      // =======================================================================
      // Story Co-Pilot Reactive State (Task 7)
      // =======================================================================
      storySessionId: 'story_' + Math.random().toString(36).substring(2, 10),
      // Setup fields: sent with the first turn only, the backend keeps them in session state.
      storyPremise: '',
      storyGenre: 'Fiction',
      customGenre: '',
      storyTone: 'Balanced',
      customTone: '',
      storyLorebookId: '',
      storyInstruction: '',
      storyInput: '',
      storyChunkSize: 1,
      storyMessages: [], // { role: 'user' | 'assistant', content, timestamp, setup? }
      isGeneratingStory: false,
      storyAbortController: null,
      storySessions: [], // past stories, newest first
      showStoryHistory: false,
    };
  },

  computed: {
    // Titles shown on the history dropdowns; a session not in the list yet is new.
    rpSessionTitle() {
      const s = this.rpSessions.find((x) => x.session_id === this.rpSessionId);
      return (s && (s.title || s.last_message)) || 'New chat';
    },
    storySessionTitle() {
      const s = this.storySessions.find((x) => x.session_id === this.storySessionId);
      return (s && (s.title || s.premise || s.session_id)) || 'New story';
    },

    // Character Filtering
    filteredCharacters() {
      const query = (this.charSearchQuery || this.charSearch || '').toLowerCase().trim();
      if (!query) {
        return this.characters;
      }
      return this.characters.filter((c) => {
        const nameMatch = (c.name || '').toLowerCase().includes(query);
        const idMatch = (c.char_id || '').toLowerCase().includes(query);
        const descMatch = (c.description || '').toLowerCase().includes(query);
        const tagMatch =
          Array.isArray(c.tags) &&
          c.tags.some((t) => (t || '').toLowerCase().includes(query));
        return nameMatch || idMatch || descMatch || tagMatch;
      });
    },

    // Lorebook Filtering
    filteredLorebooks() {
      const query = (this.lorebookSearchQuery || this.lbSearch || '').toLowerCase().trim();
      const list = Object.values(this.lorebooks || {});
      if (!query) {
        return list;
      }
      return list.filter((lb) => {
        const nameMatch = (lb.name || '').toLowerCase().includes(query);
        const descMatch = (lb.description || '').toLowerCase().includes(query);
        const idMatch = (lb.id || '').toLowerCase().includes(query);
        const entryMatch =
          Array.isArray(lb.entries) &&
          lb.entries.some((e) => {
            const contentMatch = (e.content || '').toLowerCase().includes(query);
            const keysMatch =
              Array.isArray(e.keys) &&
              e.keys.some((k) => (k || '').toLowerCase().includes(query));
            const keysStrMatch = (e.keys_str || '').toLowerCase().includes(query);
            return contentMatch || keysMatch || keysStrMatch;
          });
        return nameMatch || descMatch || idMatch || entryMatch;
      });
    },

    filteredGroups() {
      const query = (this.groupSearchQuery || '').toLowerCase().trim();
      const list = Object.values(this.groups || {});
      if (!query) return list;
      return list.filter((g) =>
        [g.name, g.group_id, g.scenario].some((s) => (s || '').toLowerCase().includes(query)),
      );
    },

    // Active Character Alternate Greetings in RP tab
    selectedCharAlternateGreetings() {
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      return char && Array.isArray(char.alternate_greetings)
        ? char.alternate_greetings
        : [];
    },

    // Active Character Display Name in RP tab
    selectedCharName() {
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      return char ? char.name : 'Assistant';
    },

    // The story so far: the Co-Pilot's replies in order.
    storyText() {
      return this.storyMessages
        .filter((m) => m.role === 'assistant')
        .map((m) => m.content.trim())
        .join('\n\n');
    },

    wordCount() {
      const text = this.storyText.trim();
      return text ? text.split(/\s+/).length : 0;
    },

    estimatedTokens() {
      return Math.round(this.wordCount * 1.3);
    },

    effectiveGenre() {
      return this.storyGenre === 'Custom' ? this.customGenre.trim() : this.storyGenre;
    },

    effectiveTone() {
      return this.storyTone === 'Custom' ? this.customTone.trim() : this.storyTone;
    },
  },

  watch: {
    charSearchQuery(newVal) {
      this.charSearch = newVal;
    },
    charSearch(newVal) {
      this.charSearchQuery = newVal;
    },
    lorebookSearchQuery(newVal) {
      this.lbSearch = newVal;
    },
    lbSearch(newVal) {
      this.lorebookSearchQuery = newVal;
    },
    charTagsInput(newVal) {
      this.charForm.tags_str = newVal;
    },
    'charForm.tags_str'(newVal) {
      if (newVal !== this.charTagsInput) {
        this.charTagsInput = newVal || '';
      }
    },
  },

  methods: {
    // =========================================================================
    // General Utilities
    // =========================================================================
    refreshIcons() {
      this.$nextTick(() => {
        if (
          typeof window !== 'undefined' &&
          window.lucide &&
          typeof window.lucide.createIcons === 'function'
        ) {
          window.lucide.createIcons();
        }
      });
    },

    showToast(message, type = 'info') {
      if (this.toastTimer) {
        clearTimeout(this.toastTimer);
      }
      this.toast = {
        show: true,
        message: String(message),
        type: type,
      };
      this.refreshIcons();
      this.toastTimer = setTimeout(() => {
        this.toast.show = false;
      }, 4000);
    },

    async checkHealth() {
      try {
        const res = await fetch('/health');
        this.backendOnline = res.ok;
      } catch (err) {
        this.backendOnline = false;
      }
    },

    renderMarkdown(text) {
      if (!text) return '';
      try {
        if (
          typeof window !== 'undefined' &&
          window.marked &&
          typeof window.marked.parse === 'function'
        ) {
          return window.marked.parse(String(text));
        }
      } catch (err) {
        console.warn('Markdown parsing error, using fallback:', err);
      }
      // Sanitized plain-text fallback with line breaks
      const escaped = String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
      return escaped.replace(/\n/g, '<br>');
    },

    // =========================================================================
    // Characters Management
    // =========================================================================
    async loadCharacters() {
      try {
        const res = await fetch('/api/v1/characters');
        if (res.ok) {
          const data = await res.json();
          let list = [];
          if (Array.isArray(data)) {
            list = data;
          } else if (data && typeof data === 'object') {
            list = Object.entries(data).map(([id, card]) => ({
              ...card,
              char_id: card.char_id || id,
            }));
          }
          this.characters = list;
          this.refreshIcons();
        }
      } catch (err) {
        console.error('Failed to load characters:', err);
      }
    },

    async selectCharacter(id) {
      if (!id) return;
      this.selectedCharId = id;
      let char = this.characters.find((c) => c.char_id === id);

      try {
        const res = await fetch(`/api/v1/characters/${encodeURIComponent(id)}`);
        if (res.ok) {
          char = await res.json();
        }
      } catch (err) {
        console.warn('Failed to fetch character detail, using cache', err);
      }

      if (char) {
        const tags = Array.isArray(char.tags) ? [...char.tags] : [];
        this.charForm = {
          char_id: char.char_id || '',
          name: char.name || '',
          tags_str: tags.join(', '),
          tags: tags,
          description: char.description || '',
          personality: char.personality || '',
          scenario: char.scenario || '',
          first_mes: char.first_mes || '',
          alternate_greetings: Array.isArray(char.alternate_greetings)
            ? [...char.alternate_greetings]
            : [],
          mes_example: char.mes_example || '',
          system_prompt: char.system_prompt || '',
          post_history_instructions: char.post_history_instructions || '',
          creator_notes: char.creator_notes || '',
          lorebook_id: char.lorebook_id || '',
        };
        this.charTagsInput = this.charForm.tags_str;
      }
      this.refreshIcons();
    },

    newCharacter() {
      this.selectedCharId = null;
      this.charForm = {
        char_id: '',
        name: '',
        tags_str: '',
        tags: [],
        description: '',
        personality: '',
        scenario: '',
        first_mes: '',
        alternate_greetings: [],
        mes_example: '',
        system_prompt: '',
        post_history_instructions: '',
        creator_notes: '',
        lorebook_id: '',
      };
      this.charTagsInput = '';
      this.refreshIcons();
    },

    addGreeting() {
      if (!this.charForm.alternate_greetings) {
        this.charForm.alternate_greetings = [];
      }
      this.charForm.alternate_greetings.push('');
      this.refreshIcons();
    },

    removeGreeting(idx) {
      if (this.charForm.alternate_greetings) {
        this.charForm.alternate_greetings.splice(idx, 1);
        this.refreshIcons();
      }
    },

    async saveCharacter() {
      const charId = (this.charForm.char_id || '').trim();
      const name = (this.charForm.name || '').trim();

      if (!charId) {
        this.showToast('Character ID (Slug) is required.', 'error');
        return;
      }
      if (!name) {
        this.showToast('Display Name is required.', 'error');
        return;
      }

      const tags = (this.charTagsInput || this.charForm.tags_str || '')
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      this.charForm.tags = tags;
      this.charForm.tags_str = tags.join(', ');

      const altGreetings = (this.charForm.alternate_greetings || [])
        .map((g) => (typeof g === 'string' ? g.trim() : ''))
        .filter(Boolean);

      const payload = {
        char_id: charId,
        name: name,
        description: this.charForm.description || '',
        personality: this.charForm.personality || '',
        scenario: this.charForm.scenario || '',
        first_mes: this.charForm.first_mes || '',
        alternate_greetings: altGreetings,
        mes_example: this.charForm.mes_example || '',
        system_prompt: this.charForm.system_prompt
          ? this.charForm.system_prompt.trim()
          : null,
        post_history_instructions: this.charForm.post_history_instructions
          ? this.charForm.post_history_instructions.trim()
          : null,
        creator_notes: this.charForm.creator_notes
          ? this.charForm.creator_notes.trim()
          : null,
        lorebook_id: this.charForm.lorebook_id || null,
        tags: tags,
      };

      try {
        const res = await fetch('/api/v1/characters', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          this.showToast(`Character "${payload.name}" saved!`, 'success');
          this.selectedCharId = payload.char_id;
          await this.loadCharacters();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to save character' }));
          this.showToast(err.detail || 'Save failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while saving character', 'error');
      }
    },

    async deleteCharacter(id) {
      const charId = id || this.selectedCharId || this.charForm.char_id;
      if (!charId) return;

      const confirmed =
        typeof confirm === 'function'
          ? confirm(`Are you sure you want to delete character "${charId}"?`)
          : true;
      if (!confirmed) return;

      try {
        const res = await fetch(`/api/v1/characters/${encodeURIComponent(charId)}`, {
          method: 'DELETE',
        });
        if (res.ok) {
          this.showToast(`Character "${charId}" deleted.`, 'success');
          if (this.selectedCharId === charId) {
            this.newCharacter();
          }
          await this.loadCharacters();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to delete' }));
          this.showToast(err.detail || 'Delete failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while deleting character', 'error');
      }
    },

    importCharacterJSON(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const raw = JSON.parse(e.target.result);
          // Support Tavern V2 spec ({ spec: 'chara_card_v2', data: { ... } }) or flat card
          const data = raw.data || raw;
          const name = data.name || 'Imported Character';
          const defaultSlug =
            name
              .toLowerCase()
              .replace(/[^a-z0-9_-]+/g, '_')
              .replace(/^_+|_+$/g, '') || 'character';
          const charId = raw.char_id || data.char_id || defaultSlug;

          let tags = [];
          if (Array.isArray(data.tags)) {
            tags = data.tags;
          } else if (typeof data.tags === 'string') {
            tags = data.tags
              .split(',')
              .map((t) => t.trim())
              .filter(Boolean);
          }

          let altGreetings = [];
          if (Array.isArray(data.alternate_greetings)) {
            altGreetings = data.alternate_greetings;
          }

          this.charForm = {
            char_id: charId,
            name: name,
            tags_str: tags.join(', '),
            tags: tags,
            description: data.description || '',
            personality: data.personality || '',
            scenario: data.scenario || '',
            first_mes: data.first_mes || '',
            alternate_greetings: altGreetings,
            mes_example: data.mes_example || '',
            system_prompt: data.system_prompt || '',
            post_history_instructions: data.post_history_instructions || '',
            creator_notes: data.creator_notes || '',
            lorebook_id: data.lorebook_id || '',
          };
          this.charTagsInput = this.charForm.tags_str;
          this.selectedCharId = charId;
          this.showToast(`Imported "${name}". Click Save to persist.`, 'info');
          this.refreshIcons();
        } catch (err) {
          this.showToast('Failed to parse character JSON file.', 'error');
        }
      };
      reader.readAsText(file);
      event.target.value = '';
    },

    exportCharacterJSON() {
      if (!this.charForm.char_id) {
        this.showToast('No character selected to export.', 'error');
        return;
      }
      const tags = (this.charTagsInput || this.charForm.tags_str || '')
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);

      const card = {
        char_id: this.charForm.char_id,
        name: this.charForm.name,
        description: this.charForm.description || '',
        personality: this.charForm.personality || '',
        scenario: this.charForm.scenario || '',
        first_mes: this.charForm.first_mes || '',
        alternate_greetings: this.charForm.alternate_greetings || [],
        mes_example: this.charForm.mes_example || '',
        system_prompt: this.charForm.system_prompt || null,
        post_history_instructions: this.charForm.post_history_instructions || null,
        creator_notes: this.charForm.creator_notes || null,
        lorebook_id: this.charForm.lorebook_id || null,
        tags: tags,
      };

      const jsonStr = JSON.stringify(card, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${this.charForm.char_id || 'character'}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.showToast(`Character "${this.charForm.name}" exported.`, 'info');
    },

    // =========================================================================
    // Lorebooks Management
    // =========================================================================
    async loadLorebooks() {
      try {
        const res = await fetch('/api/v1/lorebooks');
        if (res.ok) {
          const data = await res.json();
          const dict = {};
          if (Array.isArray(data)) {
            data.forEach((lb) => {
              const id =
                lb.id ||
                (lb.name
                  ? lb.name
                      .toLowerCase()
                      .replace(/[^a-z0-9_-]+/g, '_')
                      .replace(/^_+|_+$/g, '')
                  : 'lorebook');
              dict[id] = {
                ...lb,
                id: id,
                entries: (lb.entries || []).map((e) => ({
                  ...e,
                  keys: Array.isArray(e.keys) ? e.keys : [],
                  keys_str: Array.isArray(e.keys) ? e.keys.join(', ') : e.keys_str || '',
                  insertion_order:
                    typeof e.insertion_order === 'number' ? e.insertion_order : 100,
                  enabled: e.enabled !== false,
                })),
              };
            });
          } else if (data && typeof data === 'object') {
            for (const [id, lb] of Object.entries(data)) {
              dict[id] = {
                ...lb,
                id: lb.id || id,
                entries: (lb.entries || []).map((e) => ({
                  ...e,
                  keys: Array.isArray(e.keys) ? e.keys : [],
                  keys_str: Array.isArray(e.keys) ? e.keys.join(', ') : e.keys_str || '',
                  insertion_order:
                    typeof e.insertion_order === 'number' ? e.insertion_order : 100,
                  enabled: e.enabled !== false,
                })),
              };
            }
          }
          this.lorebooks = dict;
          this.refreshIcons();
        }
      } catch (err) {
        console.error('Failed to load lorebooks:', err);
      }
    },

    async selectLorebook(id) {
      if (!id) return;
      this.selectedLorebookId = id;
      let lb = this.lorebooks[id];

      try {
        const res = await fetch(`/api/v1/lorebooks/${encodeURIComponent(id)}`);
        if (res.ok) {
          const remote = await res.json();
          lb = { ...remote, id: id };
        }
      } catch (err) {
        console.warn('Failed to fetch lorebook detail, using cached', err);
      }

      if (lb) {
        this.lorebookForm = {
          name: lb.name || '',
          description: lb.description || '',
          entries: (lb.entries || []).map((e) => ({
            enabled: e.enabled !== false,
            insertion_order:
              typeof e.insertion_order === 'number' ? e.insertion_order : 100,
            keys: Array.isArray(e.keys) ? [...e.keys] : [],
            keys_str: Array.isArray(e.keys) ? e.keys.join(', ') : e.keys_str || '',
            content: e.content || '',
          })),
        };
      }
      this.refreshIcons();
    },

    newLorebook() {
      this.selectedLorebookId = null;
      this.lorebookForm = {
        name: '',
        description: '',
        entries: [],
      };
      this.refreshIcons();
    },

    addEntry() {
      if (!this.lorebookForm.entries) {
        this.lorebookForm.entries = [];
      }
      this.lorebookForm.entries.push({
        enabled: true,
        insertion_order: 100,
        keys: [],
        keys_str: '',
        content: '',
      });
      this.refreshIcons();
    },

    removeEntry(idx) {
      if (this.lorebookForm.entries) {
        this.lorebookForm.entries.splice(idx, 1);
        this.refreshIcons();
      }
    },

    async saveLorebook() {
      const name = (this.lorebookForm.name || '').trim();
      if (!name) {
        this.showToast('Lorebook Title is required.', 'error');
        return;
      }

      const entries = (this.lorebookForm.entries || []).map((e) => {
        let keys = [];
        if (typeof e.keys_str === 'string') {
          keys = e.keys_str
            .split(',')
            .map((k) => k.trim())
            .filter(Boolean);
        } else if (Array.isArray(e.keys)) {
          keys = e.keys.filter(Boolean);
        }
        return {
          keys: keys,
          content: e.content || '',
          insertion_order:
            typeof e.insertion_order === 'number' ? e.insertion_order : 100,
          enabled: e.enabled !== false,
        };
      });

      const payload = {
        name: name,
        description: this.lorebookForm.description || '',
        entries: entries,
      };

      try {
        const res = await fetch('/api/v1/lorebooks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const result = await res.json();
          this.showToast(`Lorebook "${payload.name}" saved!`, 'success');
          this.selectedLorebookId = result.lorebook_id;
          await this.loadLorebooks();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to save lorebook' }));
          this.showToast(err.detail || 'Save failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while saving lorebook', 'error');
      }
    },

    async deleteLorebook(id) {
      const lbId = id || this.selectedLorebookId;
      if (!lbId) return;

      const confirmed =
        typeof confirm === 'function'
          ? confirm(`Are you sure you want to delete lorebook "${lbId}"?`)
          : true;
      if (!confirmed) return;

      try {
        const res = await fetch(`/api/v1/lorebooks/${encodeURIComponent(lbId)}`, {
          method: 'DELETE',
        });
        if (res.ok) {
          this.showToast(`Lorebook "${lbId}" deleted.`, 'success');
          if (this.selectedLorebookId === lbId) {
            this.newLorebook();
          }
          await this.loadLorebooks();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to delete' }));
          this.showToast(err.detail || 'Delete failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while deleting lorebook', 'error');
      }
    },

    importLorebookJSON(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const raw = JSON.parse(e.target.result);
          const name = raw.name || raw.title || 'Imported Lorebook';
          const description = raw.description || '';
          let rawEntries = [];
          if (Array.isArray(raw.entries)) {
            rawEntries = raw.entries;
          } else if (raw.entries && typeof raw.entries === 'object') {
            rawEntries = Object.values(raw.entries);
          }

          const entries = rawEntries.map((entry) => {
            let keys = [];
            if (Array.isArray(entry.keys)) {
              keys = entry.keys;
            } else if (typeof entry.keys === 'string') {
              keys = entry.keys
                .split(',')
                .map((k) => k.trim())
                .filter(Boolean);
            } else if (Array.isArray(entry.key)) {
              keys = entry.key;
            } else if (typeof entry.key === 'string') {
              keys = entry.key
                .split(',')
                .map((k) => k.trim())
                .filter(Boolean);
            }

            return {
              enabled: entry.enabled !== false,
              insertion_order:
                typeof entry.insertion_order === 'number'
                  ? entry.insertion_order
                  : entry.order || 100,
              keys: keys,
              keys_str: keys.join(', '),
              content: entry.content || entry.text || '',
            };
          });

          this.lorebookForm = {
            name: name,
            description: description,
            entries: entries,
          };
          this.selectedLorebookId = null;
          this.showToast(`Imported "${name}". Click Save to persist.`, 'info');
          this.refreshIcons();
        } catch (err) {
          this.showToast('Failed to parse lorebook JSON file.', 'error');
        }
      };
      reader.readAsText(file);
      event.target.value = '';
    },

    exportLorebookJSON() {
      if (!this.lorebookForm.name) {
        this.showToast('No lorebook to export.', 'error');
        return;
      }

      const entries = (this.lorebookForm.entries || []).map((e) => ({
        keys:
          typeof e.keys_str === 'string'
            ? e.keys_str
                .split(',')
                .map((k) => k.trim())
                .filter(Boolean)
            : Array.isArray(e.keys)
            ? e.keys
            : [],
        content: e.content || '',
        insertion_order:
          typeof e.insertion_order === 'number' ? e.insertion_order : 100,
        enabled: e.enabled !== false,
      }));

      const payload = {
        name: this.lorebookForm.name,
        description: this.lorebookForm.description || '',
        entries: entries,
      };

      const jsonStr = JSON.stringify(payload, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${this.selectedLorebookId || 'lorebook'}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.showToast(`Lorebook "${payload.name}" exported.`, 'info');
    },

    // =========================================================================
    // Groups Management
    // =========================================================================
    async loadGroups() {
      try {
        const res = await fetch('/api/v1/groups');
        if (res.ok) this.groups = await res.json();
      } catch (err) {
        console.error('Failed to load groups:', err);
      }
      this.refreshIcons();
    },

    characterName(charId) {
      const c = this.characters.find((x) => x.char_id === charId);
      return c ? c.name : charId;
    },

    selectGroup(id) {
      const g = this.groups[id];
      if (!g) return;
      this.selectedGroupId = id;
      this.groupForm = {
        group_id: g.group_id,
        name: g.name || '',
        char_ids: [...(g.char_ids || [])],
        scenario: g.scenario || '',
        first_mes: g.first_mes || '',
        lorebook_id: g.lorebook_id || '',
      };
      this.refreshIcons();
    },

    newGroup() {
      this.selectedGroupId = null;
      this.groupForm = { group_id: '', name: '', char_ids: [], scenario: '', first_mes: '', lorebook_id: '' };
      this.refreshIcons();
    },

    // Members speak in the order they were added when the speaker selector picks no one.
    toggleGroupMember(charId) {
      const ids = this.groupForm.char_ids;
      const i = ids.indexOf(charId);
      if (i >= 0) ids.splice(i, 1);
      else ids.push(charId);
    },

    async saveGroup() {
      const groupId = (this.groupForm.group_id || '').trim();
      const name = (this.groupForm.name || '').trim();
      if (!groupId || !name) {
        this.showToast('Group ID and name are required.', 'error');
        return;
      }
      if (this.groupForm.char_ids.length === 0) {
        this.showToast('Add at least one character to the group.', 'error');
        return;
      }
      const payload = {
        group_id: groupId,
        name: name,
        char_ids: [...this.groupForm.char_ids],
        scenario: this.groupForm.scenario || '',
        first_mes: this.groupForm.first_mes || '',
        lorebook_id: this.groupForm.lorebook_id || null,
      };
      try {
        const res = await fetch('/api/v1/groups', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          this.showToast(`Group "${name}" saved!`, 'success');
          this.selectedGroupId = groupId;
          await this.loadGroups();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to save group' }));
          this.showToast(err.detail || 'Save failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while saving group', 'error');
      }
    },

    async deleteGroup(id) {
      if (!id || !confirm(`Are you sure you want to delete group "${id}"?`)) return;
      try {
        const res = await fetch(`/api/v1/groups/${encodeURIComponent(id)}`, { method: 'DELETE' });
        if (res.ok) {
          this.showToast(`Group "${id}" deleted.`, 'success');
          if (this.selectedGroupId === id) this.newGroup();
          await this.loadGroups();
        } else {
          const err = await res.json().catch(() => ({ detail: 'Failed to delete' }));
          this.showToast(err.detail || 'Delete failed', 'error');
        }
      } catch (err) {
        this.showToast('Network error while deleting group', 'error');
      }
    },

    // =========================================================================
    // Roleplay Tab Reactive Logic & SSE Streaming (Task 6)
    // =========================================================================
    scrollRPChatToBottom() {
      this.$nextTick(() => {
        if (typeof document !== 'undefined') {
          const feed = document.getElementById('rp-chat-feed');
          if (feed) {
            feed.scrollTop = feed.scrollHeight;
          }
        }
      });
    },

    onRPCharChange() {
      // Each character gets its own chat, so switching starts a new session.
      this.selectedGreetingIndex = 0;
      this.newRPSession();
      this.loadRPSessions();
    },

    onGreetingChange() {
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      if (!char) return;
      let text = char.first_mes || '';
      if (
        this.selectedGreetingIndex > 0 &&
        Array.isArray(char.alternate_greetings)
      ) {
        text =
          char.alternate_greetings[this.selectedGreetingIndex - 1] || text;
      }
      if (this.rpMessages.length === 0) {
        if (text) {
          this.rpMessages.push({
            role: 'assistant',
            content: text,
            isGreeting: true,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            }),
          });
        }
      } else if (this.rpMessages[0].role === 'assistant' && (this.rpMessages[0].isGreeting || this.rpMessages.length === 1)) {
        this.rpMessages[0].content = text;
        this.rpMessages[0].isGreeting = true;
        this.rpMessages[0].timestamp = new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        });
      }
      this.refreshIcons();
      this.scrollRPChatToBottom();
    },

    newRPSession() {
      if (this.isGeneratingRP) {
        this.stopGeneratingRP();
      }
      this.rpSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
      this.rpMessages = [];
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      this.rpLorebookId = (char && char.lorebook_id) || '';
      if (char) {
        let text = char.first_mes || '';
        if (
          this.selectedGreetingIndex > 0 &&
          Array.isArray(char.alternate_greetings)
        ) {
          text =
            char.alternate_greetings[this.selectedGreetingIndex - 1] || text;
        }
        if (text) {
          this.rpMessages.push({
            role: 'assistant',
            content: text,
            isGreeting: true,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            }),
          });
        }
      }
      this.showToast('New roleplay session initialized.', 'info');
      this.refreshIcons();
      this.scrollRPChatToBottom();
    },

    async clearRPSession() {
      if (this.isGeneratingRP) {
        this.stopGeneratingRP();
      }
      const sessionId = this.rpSessionId;
      if (sessionId) {
        try {
          const res = await fetch(`/api/v1/rp/sessions/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
          });
          if (!res.ok) {
            console.warn('DELETE session endpoint returned non-ok status:', res.status);
          }
        } catch (err) {
          console.warn('Network error while deleting session:', err);
        }
      }
      this.rpMessages = [];
      this.rpLorebookSentKey = null;
      this.showToast('Roleplay session cleared.', 'info');
      this.loadRPSessions();
      this.refreshIcons();
    },

    formatSessionTime(seconds) {
      return new Date(seconds * 1000).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    },

    toggleRPHistory() {
      this.showRPHistory = !this.showRPHistory;
      if (this.showRPHistory) this.loadRPSessions();
    },

    startRename(s) {
      this.editingSessionId = s.session_id;
      this.editingTitle = s.title || '';
      this.$nextTick(() => {
        const input = document.getElementById(`title-input-${s.session_id}`);
        if (input) input.focus();
      });
    },

    cancelRename() {
      this.editingSessionId = null;
    },

    // kind is 'rp' or 'story'. Runs on Enter and on blur; the first call wins.
    async saveRename(kind, s) {
      if (this.editingSessionId !== s.session_id) return;
      this.editingSessionId = null;
      const title = this.editingTitle.trim();
      if (title === (s.title || '')) return;
      const base = kind === 'rp' ? '/api/v1/rp/sessions' : '/api/v1/story/sessions';
      try {
        const res = await fetch(`${base}/${encodeURIComponent(s.session_id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        });
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        s.title = title;
      } catch (err) {
        this.showToast(err.message || 'Failed to rename.', 'error');
      }
    },

    async loadRPSessions() {
      if (!this.rpCharId) {
        this.rpSessions = [];
        return;
      }
      try {
        const res = await fetch(`/api/v1/rp/sessions?char_id=${encodeURIComponent(this.rpCharId)}`);
        if (res.ok) this.rpSessions = await res.json();
      } catch (err) {
        console.warn('Failed to load roleplay history:', err);
      }
      this.refreshIcons();
    },

    async openRPSession(s) {
      this.stopGeneratingRP();
      try {
        const res = await fetch(`/api/v1/rp/sessions/${encodeURIComponent(s.session_id)}/turns`);
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        const { turns } = await res.json();
        const messages = turns.map((t) => ({
          role: t.role === 'user' ? 'user' : 'assistant',
          content: t.text,
          timestamp: '',
        }));
        if (s.greeting) {
          messages.unshift({ role: 'assistant', content: s.greeting, isGreeting: true, timestamp: '' });
        }
        this.rpSessionId = s.session_id;
        this.rpMessages = messages;
        this.rpUserName = s.user_name || 'User';
        this.rpAuthorsNote = s.authors_note || '';
        this.rpLorebookId = s.lorebook_id || '';
        this.rpLorebookSentKey = null; // re-send the selected lorebook with the next message
        this.showRPHistory = false;
        this.refreshIcons();
        this.scrollRPChatToBottom();
      } catch (err) {
        this.showToast(err.message || 'Failed to open chat.', 'error');
      }
    },

    async deleteRPSession(s) {
      if (!confirm('Delete this chat? This cannot be undone.')) return;
      if (s.session_id === this.rpSessionId) {
        await this.clearRPSession();
        this.newRPSession();
      } else {
        await fetch(`/api/v1/rp/sessions/${encodeURIComponent(s.session_id)}`, { method: 'DELETE' });
        this.loadRPSessions();
      }
    },

    async copyMessage(text) {
      const content = text || '';
      try {
        if (
          typeof navigator !== 'undefined' &&
          navigator.clipboard &&
          typeof navigator.clipboard.writeText === 'function'
        ) {
          await navigator.clipboard.writeText(content);
          this.showToast('Message copied to clipboard.', 'info');
          return;
        }
      } catch (err) {
        console.warn('Clipboard API writeText failed, using fallback:', err);
      }
      // Fallback for non-secure contexts or restricted clipboard permissions
      try {
        const el = document.createElement('textarea');
        el.value = content;
        el.setAttribute('readonly', '');
        el.style.position = 'fixed';
        el.style.opacity = '0';
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
        this.showToast('Message copied to clipboard.', 'info');
      } catch (err) {
        this.showToast('Failed to copy message to clipboard.', 'error');
      }
    },

    stopGeneratingRP() {
      if (this.rpAbortController) {
        try {
          this.rpAbortController.abort();
        } catch (_) {}
        this.rpAbortController = null;
      }
      this.isGeneratingRP = false;
      this.refreshIcons();
    },

    async deleteTurn(index) {
      if (index === undefined || index === null || index < 0 || index >= this.rpMessages.length) {
        return;
      }

      const hasGreeting = this.rpMessages.length > 0 && !!this.rpMessages[0].isGreeting;

      // When deleting index 0 and it's the unpersisted greeting: client-side removal only
      if (index === 0 && hasGreeting) {
        this.rpMessages.splice(0, 1);
        if (this.rpMessages.length === 0) {
          await this.clearRPSession();
        } else {
          this.showToast('Turn deleted.', 'info');
          this.refreshIcons();
        }
        return;
      }

      const backendIndex = hasGreeting ? index - 1 : index;
      const sessionId = this.rpSessionId;
      if (sessionId && backendIndex >= 0) {
        try {
          const res = await fetch(`/api/v1/rp/sessions/${encodeURIComponent(sessionId)}/turns/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              turn_index: backendIndex,
              truncate_subsequent: false,
            }),
          });
          if (!res.ok) {
            console.warn('Backend turn delete response not ok:', res.status);
          }
        } catch (err) {
          console.warn('Failed to delete turn on backend:', err);
        }
      }
      this.rpMessages.splice(index, 1);
      this.showToast('Turn deleted.', 'info');
      this.refreshIcons();
    },

    async deleteFromHere(index) {
      if (index === undefined || index === null || index < 0 || index >= this.rpMessages.length) {
        return;
      }
      const confirmed =
        typeof confirm === 'function'
          ? confirm('Rewind and delete all messages from this turn forward?')
          : true;
      if (!confirmed) return;

      if (this.isGeneratingRP) {
        this.stopGeneratingRP();
      }

      const hasGreeting = this.rpMessages.length > 0 && !!this.rpMessages[0].isGreeting;

      // If rewinding from index 0, clear the whole session
      if (index === 0) {
        await this.clearRPSession();
        return;
      }

      const backendIndex = hasGreeting ? index - 1 : index;
      const sessionId = this.rpSessionId;
      if (sessionId && backendIndex >= 0) {
        try {
          const res = await fetch(`/api/v1/rp/sessions/${encodeURIComponent(sessionId)}/turns/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              turn_index: backendIndex,
              truncate_subsequent: true,
            }),
          });
          if (!res.ok) {
            console.warn('Backend deleteFromHere response not ok:', res.status);
          }
        } catch (err) {
          console.warn('Failed to rewind session on backend:', err);
        }
      }
      this.rpMessages.splice(index);
      this.showToast('Rewound chat history.', 'info');
      this.refreshIcons();
    },

    async regenerateTurn(index) {
      if (this.isGeneratingRP) return;
      const targetIdx = typeof index === 'number' ? index : this.rpMessages.length - 1;
      if (targetIdx < 0 || targetIdx >= this.rpMessages.length) return;

      const targetMsg = this.rpMessages[targetIdx];
      if (targetMsg.role !== 'assistant') {
        this.showToast('Can only regenerate assistant replies.', 'info');
        return;
      }

      // Find prior user prompt
      let userIdx = -1;
      let priorUserText = '';
      for (let i = targetIdx - 1; i >= 0; i--) {
        if (this.rpMessages[i].role === 'user') {
          userIdx = i;
          priorUserText = this.rpMessages[i].content;
          break;
        }
      }

      if (!priorUserText || userIdx === -1) {
        this.showToast('No prior user message found to regenerate from.', 'error');
        return;
      }

      const hasGreeting = this.rpMessages.length > 0 && !!this.rpMessages[0].isGreeting;
      const userBackendIndex = hasGreeting ? userIdx - 1 : userIdx;

      // Rewind backend session to before the prior user message so re-sending it won't duplicate it in memory
      const sessionId = this.rpSessionId;
      if (sessionId && userBackendIndex >= 0) {
        try {
          await fetch(`/api/v1/rp/sessions/${encodeURIComponent(sessionId)}/turns/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              turn_index: userBackendIndex,
              truncate_subsequent: true,
            }),
          });
        } catch (err) {
          console.warn('Backend rewind for regenerate failed:', err);
        }
      }

      // Remove the assistant message from the frontend array
      this.rpMessages.splice(targetIdx, 1);

      // Trigger streaming generation from the prior user message
      await this._streamAssistantReply(priorUserText);
    },

    async sendRPMessage() {
      if (this.isGeneratingRP) return;
      const text = (this.rpInput || '').trim();
      if (!text) return;
      if (!this.rpCharId) {
        this.showToast('Please select a character first.', 'error');
        return;
      }

      this.rpInput = '';
      this.rpMessages.push({
        role: 'user',
        content: text,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      });

      await this._streamAssistantReply(text);
    },

    async _streamAssistantReply(promptText) {
      const assistantMsg = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      };
      this.rpMessages.push(assistantMsg);
      this.isGeneratingRP = true;
      this.rpAbortController = new AbortController();
      this.scrollRPChatToBottom();
      this.refreshIcons();

      if (!this.rpSessionId) {
        this.rpSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
      }

      const firstMsg = this.rpMessages[0];
      const payload = {
        char_id: this.rpCharId,
        session_id: this.rpSessionId,
        message: promptText,
        // The greeting lives only in the UI, so the backend gets it with each message.
        greeting: firstMsg && firstMsg.isGreeting ? firstMsg.content : null,
        authors_note: this.rpAuthorsNote ? this.rpAuthorsNote.trim() : null,
        user_name: this.rpUserName ? this.rpUserName.trim() : 'User',
        chunk_size: Number(this.rpChunkSize) || 1,
      };
      // The backend keeps the lorebook in the session, so only send it when the
      // session or the selection changed ('' clears it).
      const lorebookKey = `${this.rpSessionId}|${this.rpLorebookId || ''}`;
      if (lorebookKey !== this.rpLorebookSentKey) {
        payload.lorebook_id = this.rpLorebookId || '';
      }

      try {
        const res = await fetch('/api/v1/rp/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: this.rpAbortController.signal,
        });

        if (!res.ok) {
          let errDetail = `Server returned HTTP ${res.status}`;
          try {
            const errJson = await res.json();
            if (errJson.detail) errDetail = errJson.detail;
          } catch (_) {}
          throw new Error(errDetail);
        }

        if (!res.body) {
          throw new Error('ReadableStream not supported on response');
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          buffer = buffer.replace(/\r\n/g, '\n');

          const events = buffer.split('\n\n');
          buffer = events.pop();

          for (const block of events) {
            const trimmed = block.trim();
            if (!trimmed) continue;
            for (const line of trimmed.split('\n')) {
              const l = line.trim();
              if (l.startsWith('data:')) {
                const rawData = l.slice(5).trim();
                if (rawData === '[DONE]') {
                  continue;
                }
                let data;
                try {
                  data = JSON.parse(rawData);
                } catch (jsonErr) {
                  continue; // Ignore malformed chunk
                }
                if (data.error) throw new Error(data.error);
                if (data.delta) {
                  assistantMsg.content += data.delta;
                  this.scrollRPChatToBottom();
                } else if (data.full_text && !assistantMsg.content) {
                  assistantMsg.content = data.full_text;
                  this.scrollRPChatToBottom();
                }
              }
            }
          }
        }

        // Flush any remaining buffered SSE lines
        if (buffer && buffer.trim()) {
          for (const line of buffer.trim().split('\n')) {
            const l = line.trim();
            if (l.startsWith('data:')) {
              const rawData = l.slice(5).trim();
              if (rawData !== '[DONE]') {
                try {
                  const data = JSON.parse(rawData);
                  if (data.delta) {
                    assistantMsg.content += data.delta;
                  } else if (data.full_text && !assistantMsg.content) {
                    assistantMsg.content = data.full_text;
                  }
                } catch (_) {}
              }
            }
          }
        }

        // Marked only after a completed turn; re-sending on a failed one is harmless.
        this.rpLorebookSentKey = lorebookKey;

        if (!assistantMsg.content.trim()) {
          assistantMsg.content = '*(No response)*';
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          this.showToast('Generation stopped.', 'info');
          if (!assistantMsg.content) {
            assistantMsg.content = '*(Generation stopped)*';
          }
        } else {
          console.error('Roleplay streaming error:', err);
          this.showToast(err.message || 'Error generating roleplay response.', 'error');
          if (!assistantMsg.content) {
            assistantMsg.content = `*(Error: ${err.message || 'Failed to generate response'})*`;
          }
        }
      } finally {
        this.isGeneratingRP = false;
        this.rpAbortController = null;
        this.loadRPSessions();
        this.scrollRPChatToBottom();
        this.refreshIcons();
      }
    },

    // =========================================================================
    // Story Co-Pilot Chat: a setup card starts the story, then it is a chat
    // =========================================================================
    scrollStoryChatToBottom() {
      this.$nextTick(() => {
        const feed = document.getElementById('story-chat-feed');
        if (feed) feed.scrollTop = feed.scrollHeight;
      });
    },

    newStorySession() {
      this.stopGeneratingStory();
      this.storySessionId = 'story_' + Math.random().toString(36).substring(2, 10);
      this.storyMessages = [];
      this.storyInput = '';
      this.showToast('New story session initialized.', 'info');
    },

    async loadStorySessions() {
      try {
        const res = await fetch('/api/v1/story/sessions');
        if (res.ok) this.storySessions = await res.json();
      } catch (err) {
        console.warn('Failed to load story history:', err);
      }
      this.refreshIcons();
    },

    toggleStoryHistory() {
      this.showStoryHistory = !this.showStoryHistory;
      if (this.showStoryHistory) this.loadStorySessions();
    },

    async openStorySession(s) {
      this.stopGeneratingStory();
      try {
        const res = await fetch(`/api/v1/story/sessions/${encodeURIComponent(s.session_id)}/messages`);
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        const { messages, setup } = await res.json();
        this.storyMessages = messages.map((m) => ({ ...m, timestamp: '' }));
        if (this.storyMessages.length > 0) this.storyMessages[0].setup = setup;
        this.storySessionId = s.session_id;
        this.storyInput = '';
        this.showStoryHistory = false;
        this.scrollStoryChatToBottom();
      } catch (err) {
        this.showToast(err.message || 'Failed to open story.', 'error');
      }
    },

    async deleteStorySession(s) {
      if (!confirm('Delete this story? This cannot be undone.')) return;
      await fetch(`/api/v1/story/sessions/${encodeURIComponent(s.session_id)}`, { method: 'DELETE' });
      if (s.session_id === this.storySessionId) {
        this.stopGeneratingStory();
        this.storySessionId = 'story_' + Math.random().toString(36).substring(2, 10);
        this.storyMessages = [];
      }
      this.loadStorySessions();
    },

    async startStory() {
      if (this.isGeneratingStory) return;
      const instruction = this.storyInstruction.trim() || 'Begin the story.';
      const setup = {
        premise: this.storyPremise.trim(),
        genre: this.effectiveGenre || 'Fiction',
        tone: this.effectiveTone || 'Balanced',
      };
      this.storyMessages.push({ role: 'user', content: instruction, setup, timestamp: this._timeNow() });
      await this._streamStoryReply({ ...setup, lorebook_id: this.storyLorebookId || null, instruction });
    },

    async sendStoryMessage() {
      if (this.isGeneratingStory) return;
      const text = this.storyInput.trim();
      if (!text) return;
      this.storyInput = '';
      this.storyMessages.push({ role: 'user', content: text, timestamp: this._timeNow() });
      await this._streamStoryReply({ instruction: text });
    },

    _timeNow() {
      return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    async _streamStoryReply(fields) {
      this.storyMessages.push({ role: 'assistant', content: '', timestamp: this._timeNow() });
      // Read it back through the reactive array so streamed text re-renders.
      const reply = this.storyMessages[this.storyMessages.length - 1];
      this.isGeneratingStory = true;
      this.storyAbortController = new AbortController();
      this.scrollStoryChatToBottom();

      try {
        const res = await fetch('/api/v1/story/expand/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: this.storySessionId,
            chunk_size: Number(this.storyChunkSize) || 1,
            ...fields,
          }),
          signal: this.storyAbortController.signal,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `Server returned HTTP ${res.status}`);
        }

        // Each SSE event is one `data:` line: {"delta"}, then {"full_text", "done"} (or {"error"}), then [DONE].
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const raw = line.slice(5).trim();
            if (raw === '[DONE]') continue;
            const data = JSON.parse(raw);
            if (data.error) throw new Error(data.error);
            if (data.delta) {
              reply.content += data.delta;
            } else if (data.full_text && !reply.content) {
              reply.content = data.full_text;
            }
            this.scrollStoryChatToBottom();
          }
        }
        if (!reply.content.trim()) reply.content = '*(No response)*';
      } catch (err) {
        if (err.name === 'AbortError') {
          this.showToast('Story generation stopped.', 'info');
          if (!reply.content) reply.content = '*(Generation stopped)*';
        } else {
          console.error('Story generation error:', err);
          this.showToast(err.message || 'Error generating story.', 'error');
          if (!reply.content) reply.content = `*(Error: ${err.message || 'Failed to generate story'})*`;
        }
      } finally {
        this.isGeneratingStory = false;
        this.storyAbortController = null;
        this.loadStorySessions();
        this.scrollStoryChatToBottom();
      }
    },

    stopGeneratingStory() {
      if (this.storyAbortController) this.storyAbortController.abort();
      this.storyAbortController = null;
      this.isGeneratingStory = false;
    },

    exportStory(format) {
      const ext = format === 'md' ? 'md' : 'txt';
      const blob = new Blob([this.storyText], { type: ext === 'md' ? 'text/markdown' : 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${this.storySessionId || 'story'}.${ext}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.showToast(`Story exported as .${ext}`, 'info');
    },
  },

  updated() {
    if (
      typeof window !== 'undefined' &&
      window.lucide &&
      typeof window.lucide.createIcons === 'function'
    ) {
      window.lucide.createIcons();
    }
  },

  mounted() {
    this.checkHealth();
    this.healthInterval = setInterval(() => this.checkHealth(), 10000);
    this.loadCharacters();
    this.loadLorebooks();
    this.loadGroups();
    this.loadStorySessions();
    this.refreshIcons();
  },

  beforeUnmount() {
    if (this.healthInterval) {
      clearInterval(this.healthInterval);
    }
  },
};

// Mount Vue application if Vue is loaded in browser
if (typeof Vue !== 'undefined' && Vue && typeof Vue.createApp === 'function') {
  const vueApp = Vue.createApp(AppDefinition);
  if (typeof window !== 'undefined') {
    window.app = vueApp.mount('#app');
  }
} else if (
  typeof window !== 'undefined' &&
  window.Vue &&
  typeof window.Vue.createApp === 'function'
) {
  const vueApp = window.Vue.createApp(AppDefinition);
  window.app = vueApp.mount('#app');
}

// Export module definition if running in module/test environment
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AppDefinition;
}
