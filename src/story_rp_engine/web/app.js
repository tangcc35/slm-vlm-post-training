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
      activeTab: 'roleplay', // 'roleplay' | 'story' | 'characters' | 'lorebooks'
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
      // Roleplay Tab Reactive Defaults & Stubs (Extended in Task 6)
      // =======================================================================
      rpCharId: '',
      selectedGreetingIndex: 0,
      rpSessionId: 'rp_session_1',
      rpUserName: 'User',
      rpLorebookId: '',
      rpAuthorsNote: '',
      rpChunkSize: 16,
      rpMessages: [],
      rpInput: '',
      isGeneratingRP: false,

      // =======================================================================
      // Story Co-Pilot Reactive Defaults & Stubs (Extended in Task 7)
      // =======================================================================
      storySessionId: 'story_session_1',
      storyPremise: '',
      storyGenre: 'Fiction',
      customGenre: '',
      storyTone: 'Balanced',
      customTone: '',
      storyInstruction: 'Continue the story naturally from the current point.',
      storyMaxTokens: 512,
      storyChunkSize: 16,
      storyCurrentText: '',
      storyHistory: [],
      showDirectorBeats: false,
      directorBeats: [],
      isGeneratingStory: false,
    };
  },

  computed: {
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

    // Story Word & Token Counters
    wordCount() {
      const text = (this.storyCurrentText || '').trim();
      return text ? text.split(/\s+/).length : 0;
    },

    estimatedTokens() {
      return Math.round(this.wordCount * 1.3);
    },

    canUndo() {
      return Array.isArray(this.storyHistory) && this.storyHistory.length > 0;
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

    renderMarkdown(content) {
      if (
        typeof window !== 'undefined' &&
        window.marked &&
        typeof window.marked.parse === 'function'
      ) {
        return window.marked.parse(content || '');
      }
      return (content || '').replace(/\n/g, '<br>');
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
        keys: (e.keys_str || '')
          .split(',')
          .map((k) => k.trim())
          .filter(Boolean),
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
    // Roleplay Tab Defaults & Stubs (Extended in Task 6)
    // =========================================================================
    onRPCharChange() {
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      this.selectedGreetingIndex = 0;
      if (char && char.first_mes && this.rpMessages.length === 0) {
        this.rpMessages.push({
          role: 'assistant',
          content: char.first_mes,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        });
      }
      this.refreshIcons();
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
      if (this.rpMessages.length <= 1) {
        this.rpMessages = [
          {
            role: 'assistant',
            content: text,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            }),
          },
        ];
      }
      this.refreshIcons();
    },

    newRPSession() {
      this.rpSessionId = 'rp_' + Date.now().toString(36);
      this.rpMessages = [];
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      if (char && char.first_mes) {
        this.rpMessages.push({
          role: 'assistant',
          content: char.first_mes,
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        });
      }
      this.showToast('New roleplay session initialized.', 'info');
      this.refreshIcons();
    },

    clearRPSession() {
      this.rpMessages = [];
      this.showToast('Roleplay session cleared.', 'info');
      this.refreshIcons();
    },

    copyMessage(content) {
      if (typeof navigator !== 'undefined' && navigator.clipboard) {
        navigator.clipboard.writeText(content || '').then(() => {
          this.showToast('Message copied to clipboard.', 'info');
        });
      }
    },

    regenerateTurn(idx) {
      this.showToast('Regenerate reply will be active with stream runner.', 'info');
    },

    deleteFromHere(idx) {
      const confirmed =
        typeof confirm === 'function'
          ? confirm('Rewind and delete all messages from this turn forward?')
          : true;
      if (confirmed) {
        this.rpMessages.splice(idx);
        this.showToast('Rewound chat history.', 'info');
        this.refreshIcons();
      }
    },

    deleteTurn(idx) {
      this.rpMessages.splice(idx, 1);
      this.refreshIcons();
    },

    sendRPMessage() {
      if (!this.rpInput || !this.rpInput.trim() || !this.rpCharId) return;
      const text = this.rpInput.trim();
      this.rpInput = '';
      this.rpMessages.push({
        role: 'user',
        content: text,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      });
      this.refreshIcons();
    },

    stopGeneratingRP() {
      this.isGeneratingRP = false;
    },

    // =========================================================================
    // Story Co-Pilot Tab Defaults & Stubs (Extended in Task 7)
    // =========================================================================
    newStorySession() {
      this.storySessionId = 'story_' + Date.now().toString(36);
      this.storyCurrentText = '';
      this.storyHistory = [];
      this.directorBeats = [];
      this.showToast('New story session initialized.', 'info');
      this.refreshIcons();
    },

    expandStory() {
      this.showToast('Story expansion stream will be active in Task 7.', 'info');
    },

    stopGeneratingStory() {
      this.isGeneratingStory = false;
    },

    undoLastExpansion() {
      if (this.canUndo) {
        this.storyCurrentText = this.storyHistory.pop();
        this.showToast('Reverted to previous expansion revision.', 'info');
      }
    },

    copyStoryDraft() {
      if (typeof navigator !== 'undefined' && navigator.clipboard) {
        navigator.clipboard.writeText(this.storyCurrentText || '').then(() => {
          this.showToast('Story draft copied to clipboard.', 'info');
        });
      }
    },

    exportStory(format) {
      const text = this.storyCurrentText || '';
      const mimeType = format === 'md' ? 'text/markdown' : 'text/plain';
      const ext = format === 'md' ? 'md' : 'txt';
      const blob = new Blob([text], { type: mimeType });
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
