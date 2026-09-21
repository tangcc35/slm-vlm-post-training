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
      // Roleplay Tab Reactive State (Task 6)
      // =======================================================================
      rpSessionId: 'sess_' + Math.random().toString(36).substring(2, 10),
      rpCharId: '',
      rpUserName: 'User',
      rpAuthorsNote: '',
      rpLorebookId: '',
      rpChunkSize: 16,
      rpMessages: [],
      selectedGreetingIndex: 0,
      isGeneratingRP: false,
      rpAbortController: null,
      rpInput: '',

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
      const char = this.characters.find((c) => c.char_id === this.rpCharId);
      this.selectedGreetingIndex = 0;
      if (char) {
        const text = char.first_mes || '';
        // If chat is empty, or only contains an initial assistant greeting, set/update greeting
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
        } else if (this.rpMessages.length === 1 && this.rpMessages[0].role === 'assistant') {
          if (text) {
            this.rpMessages[0].content = text;
            this.rpMessages[0].isGreeting = true;
            this.rpMessages[0].timestamp = new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            });
          } else {
            this.rpMessages = [];
          }
        }
      }
      this.refreshIcons();
      this.scrollRPChatToBottom();
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
      this.showToast('Roleplay session cleared.', 'info');
      this.refreshIcons();
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

      const payload = {
        char_id: this.rpCharId,
        session_id: this.rpSessionId,
        message: promptText,
        authors_note: this.rpAuthorsNote ? this.rpAuthorsNote.trim() : null,
        user_name: this.rpUserName ? this.rpUserName.trim() : 'User',
        chunk_size: Number(this.rpChunkSize) || 16,
      };

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
                try {
                  const data = JSON.parse(rawData);
                  if (data.delta) {
                    assistantMsg.content += data.delta;
                    this.scrollRPChatToBottom();
                  } else if (data.full_text && !assistantMsg.content) {
                    assistantMsg.content = data.full_text;
                    this.scrollRPChatToBottom();
                  }
                } catch (jsonErr) {
                  // Ignore malformed chunk
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
        this.scrollRPChatToBottom();
        this.refreshIcons();
      }
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
