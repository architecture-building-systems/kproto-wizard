cea_wizard/
│
├── main.py                             # Entry point: handles page routing
│
├── home_page.py                        # Simple homepage
├── userdata_page.py                    # Lists user-created sessions
│
├── kprototyper/
│   ├── __init__.py
│   ├── session.py                      # KPrototyperSession class
│   ├── logic.py                        # All backend logic (clustering, preprocessing)
│   └── ui.py                           # Contains kprototyper_main() with UI layout and navigation
│
├── archetyper/
│   ├── __init__.py
│   ├── session.py                      # ArchetyperSession class
│   ├── logic.py                        # All transformation/export logic
│   └── ui.py                           # Contains archetyper_main() with UI layout
│
├── resources_page.py                   # Optional: links, documentation
├── about_page.py                       # Optional: app info
│
├── utils/
│   ├── file_io.py                      # Shared file helpers (e.g., download, read/write)
│   ├── session_manager.py              # Shared session tracking utilities
│   └── constants.py                    # File paths, cluster color palettes, etc.
