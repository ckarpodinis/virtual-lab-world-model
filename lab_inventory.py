import os
import json
import copy
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt

INVENTORY_FILE = "inventory.json"

COMPONENT_TYPES = {
    "binary control": ["toggle", "open", "close", "set"],
    "momentary control": ["press"],
    "continuous control": ["set", "increase", "decrease"],
    "selector control": ["select"],
    "receptor": ["place_object", "remove_object"],
    "actuator": ["activate", "deactivate"],
    "observable": ["read"]
}

COMPONENT_STATE_SPECS = {
    "binary control": {
        "kind": "enum",
        "values": [0, 1]
    },
    #"momentary control": {
    #    "kind": "enum",
    #    "values": ["released", "pressed"]
    #},
    #"receptor": {
    #    "kind": "enum",
    #    "values": ["empty", "occupied"]
    #}
    # others configured by user in UI
}

# -----------------------------
# Persistence
# -----------------------------
def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    # Ensure structure
    data.setdefault("objects", {
        "instrument": {},
        "container": {},
        "tool": {},
        "material": {}
    })

    data.setdefault("interactions", [])

    return data

def save_inventory():
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(world, f, indent=2)

world = load_inventory()
inventory = world["objects"]
interactions = world["interactions"]

# Ensure fields exist
for cat in inventory:
    for tname in inventory[cat]:
        for item in inventory[cat][tname]:
                if cat != "material":
                    item.setdefault("components", [])
                    item.setdefault("object_actions", [])
                    item.setdefault("states", {})  # ensure object-level states always exists
                
                # container-specific cleanup
                if cat == "container":
                    # remove legacy key if present
                    item.pop("material", None)
                    # remove contains if explicitly null
                    if "contains" in item and item["contains"] is None:
                        item.pop("contains", None)

                # component normalization (only for non-material)
                for comp in item.get("components", []):
                    comp.setdefault("actions", [])
                    # NEW: states (state specification only)
                    if "states" not in comp:
                        ctype = comp.get("type")
                        if ctype in COMPONENT_STATE_SPECS:
                            comp["states"] = copy.deepcopy(COMPONENT_STATE_SPECS[ctype])
                        else:
                            comp["states"] = None  # will be set via GUI for selector/continuous/observable

def parse_object_label(label):
    # format: category:type[id]
    cat_part, rest = label.split(":", 1)
    type_part, id_part = rest.split("[")
    obj_id = int(id_part[:-1])

    return {
        "category": cat_part,
        "type": type_part,
        "id": obj_id
    }

# -----------------------------
# GUI
# -----------------------------
class LabInventoryApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Virtual Lab Inventory")
        self.geometry("950x650")

        self._build_ui()
        self.refresh_tree()
        self.refresh_json()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.current_right_panel_filename = "inventory.json"

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self):

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # -----------------------------
        # LEFT PANEL (Controls + Tree)
        # -----------------------------
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=2)  # larger weight → wider

        control_frame = ttk.Frame(left_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Add Item", command=self.open_item_editor_new).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Add Component", command=self.add_component).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Edit Selected", command=self.edit_selected).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Delete Selected", command=self.delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Clone Selected", command=self.clone_selected).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Add Interaction", command=self.open_interaction_editor).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Generate MDP Template", command=self.generate_selected_mdp).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Generate All MDP Templates", command=self.generate_all_mdp).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Visualize Graph", command=self.visualize_graph).pack(fill=tk.X, pady=5)

        self.tree = ttk.Treeview(left_frame, show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<Double-1>", self.on_double_click)

        # -----------------------------
        # RIGHT PANEL (JSON Preview)
        # -----------------------------
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)  # smaller weight → narrower

        ttk.Label(right_frame, text="JSON Viewer").pack(anchor=tk.W)

        # Toolbar for the JSON viewer
        json_toolbar = ttk.Frame(right_frame)
        json_toolbar.pack(fill=tk.X, pady=(2, 4))

        ttk.Button( json_toolbar, text="Show Inventory", command=self.show_inventory_json).pack(side=tk.LEFT, padx=2)
        ttk.Button( json_toolbar, text="Save JSON", command=self.save_right_panel_json).pack(side=tk.LEFT, padx=2)

        # JSON text container
        json_container = ttk.Frame(right_frame)
        json_container.pack(fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(json_container)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.json_text = tk.Text(json_container, yscrollcommand=scroll.set, width=40)
        self.json_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll.config(command=self.json_text.yview)

    def on_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return

        values = self.tree.item(sel[0], "values")
        if not values:
            return

        if values[0] == "item":
            self.open_item_editor(values[1], values[2], int(values[3]))
        elif values[0] == "component":
            self.open_component_editor(values[1], values[2], int(values[3]), int(values[4]))
        elif values[0] == "interaction":
            self.open_interaction_editor(int(values[1]))

    def get_receptor_domain(self, cat, type_name, obj_id, comp_name):
            """
            Returns domain for a receptor component:
            { "empty" } ∪ eligible objects from place interactions.
            """
            domain = ["empty"]
            for inter in interactions:
                if inter.get("type") != "place":
                    continue
                tgt = inter["target"]
                if tgt["category"] != cat:
                    continue
                if tgt["type"] != type_name:
                    continue
                if tgt.get("component") != comp_name:
                    continue
                src = inter["source"]
                # wildcard source
                if src["id"] == "*":
                    label = f"{src['category']}:{src['type']}"
                    domain.append(label)
                else:
                    label = f"{src['category']}:{src['type']}[{src['id']}]"
                    domain.append(label)
            return sorted(set(domain))

    # -----------------------------
    # ITEM EDITOR (Unified Add/Edit)
    # -----------------------------
    def open_item_editor_new(self):
        self.open_item_editor(cat=None, type_name=None, idx=None)

    def open_item_editor(self, cat, type_name, idx):
        is_edit = cat is not None

        win = tk.Toplevel(self)
        win.title("Edit Item" if is_edit else "Add Item")
        win.resizable(False, False)

        row = 0

        # -----------------------------
        # Category
        # -----------------------------
        ttk.Label(win, text="Category *").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        cat_cb = ttk.Combobox(
            win,
            values=["instrument", "tool", "container", "material"],
            state="readonly",
            width=35
        )
        cat_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # -----------------------------
        # Type Name
        # -----------------------------
        ttk.Label(win, text="Type name *").grid(row=row, column=0, sticky="w", padx=5)
        type_entry = ttk.Entry(win, width=40)
        type_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # -----------------------------
        # Object-Level Actions
        # -----------------------------
        ttk.Label(win, text="Object Actions").grid(row=row, column=0, padx=5, sticky="w")
        ttk.Label(win, text="(comma-separated)", foreground="gray").grid(row=row, column=1, sticky="w", padx=5)
        row += 1
        action_entry = ttk.Entry(win, width=40)
        action_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # -----------------------------
        # Object Attributes (NEW)
        # -----------------------------
        attr_frame = ttk.LabelFrame(win, text="Object Attributes")

        movable_var = tk.BooleanVar(value=False)
        container_var = tk.BooleanVar(value=False)

        movable_cb = ttk.Checkbutton(attr_frame, text="Movable", variable=movable_var)
        container_cb = ttk.Checkbutton(attr_frame, text="Material Container", variable=container_var)

        movable_cb.pack(anchor="w", padx=5, pady=2)
        container_cb.pack(anchor="w", padx=5, pady=2)

        attr_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        row += 1

        # -----------------------------
        # Material (Container Only)
        # -----------------------------
        material_label = ttk.Label(win, text="Material (optional)")
        material_entry = ttk.Entry(win, width=40)

        def update_material_visibility(event=None):
            if cat_cb.get() == "container":
                material_label.grid(row=row, column=0, sticky="w", padx=5)
                material_entry.grid(row=row, column=1, padx=5, pady=5)
            else:
                material_label.grid_remove()
                material_entry.grid_remove()

        cat_cb.bind("<<ComboboxSelected>>", update_material_visibility)

        # -----------------------------
        # Edit Mode Prefill
        # -----------------------------
        if is_edit:
            cat_cb.set(cat)
            cat_cb.config(state="disabled")
            type_entry.insert(0, type_name)

            item = inventory[cat][type_name][idx]

            existing_actions = item.get("object_actions", [])
            action_entry.insert(0, ", ".join(existing_actions))

            states = item.get("states", {})
            movable_var.set("location" in states)
            container_var.set("quantity" in states)

            # Prefill container material
            if cat == "container":
                contains = item.get("contains")
                if contains:
                    material_entry.insert(0, contains.get("type_name", ""))

        update_material_visibility()
        row += 1

        # -----------------------------
        # Confirm Logic
        # -----------------------------
        def confirm():
            category = cat_cb.get()

            if not category:
                messagebox.showerror("Error", "Category must be selected.")
                return

            new_type = type_entry.get().strip()
            if not new_type:
                messagebox.showerror("Error", "Type name required.")
                return

            raw_actions = action_entry.get().strip()
            selected_object_actions = [a.strip() for a in raw_actions.split(",") if a.strip()]

            material_name = material_entry.get().strip() if category == "container" else None

            # -------------------------
            # EDIT MODE
            # -------------------------
            if is_edit:
                if new_type != type_name:
                    item = inventory[cat][type_name].pop(idx)

                    for i, it in enumerate(inventory[cat][type_name]):
                        it["id"] = i

                    inventory[category].setdefault(new_type, [])
                    item["id"] = len(inventory[category][new_type])
                    inventory[category][new_type].append(item)

                    focus = ("item", category, new_type, str(item["id"]))
                else:
                    item = inventory[cat][type_name][idx]
                    focus = ("item", cat, type_name, str(idx))

                if category != "material":
                    item["object_actions"] = selected_object_actions
                    # Ensure states exists
                    item.setdefault("states", {})

                    # Movable → location state
                    if movable_var.get():
                        item["states"]["location"] = {
                            "kind": "enum",
                            "domain": "locations"
                        }
                    else:
                        item["states"].pop("location", None)

                    # Contains material → quantity + material states
                    if container_var.get():
                        item["states"]["quantity"] = {
                            "kind": "numeric",
                            "min": 0,
                            "max": 100
                        }
                        item["states"]["material"] = {
                            "kind": "enum",
                            "domain": "materials"
                        }
                    else:
                        item["states"].pop("quantity", None)
                        item["states"].pop("material", None)
                
                if category == "container":
                    if material_name:
                        item["contains"] = {
                            "entity_type": "material",
                            "type_name": material_name
                        }
                    else:
                        item.pop("contains", None)

            # -------------------------
            # ADD MODE
            # -------------------------
            else:
                inventory[category].setdefault(new_type, [])
                new_item_id = len(inventory[category][new_type])

                if category == "material":
                    new_item = {
                        "id": new_item_id
                    }
                else:
                    new_item = {
                        "id": new_item_id,
                        "components": [],
                        "object_actions": selected_object_actions,
                        "states": {}
                    }

                    if movable_var.get():
                        new_item["states"]["location"] = {
                            "kind": "enum",
                            "domain": "locations"
                        }

                    if container_var.get():
                        new_item["states"]["quantity"] = {
                            "kind": "numeric",
                            "min": 0,
                            "max": 100
                        }
                if category == "container" and material_name:
                    new_item["contains"] = {
                        "entity_type": "material",
                        "type_name": material_name
                    }

                inventory[category][new_type].append(new_item)

                focus = ("item", category, new_type, str(new_item_id))

            win.destroy()
            self.after_change(focus_values=focus)

        # -----------------------------
        # Save Button
        # -----------------------------
        ttk.Button(
            win,
            text="Save" if is_edit else "Add",
            command=confirm
        ).grid(row=row, column=1, pady=15, sticky="e")

        self.center_window(win)
        win.grab_set()

    # -----------------------------
    # COMPONENT EDITOR (Structured Actions)
    # -----------------------------
    def open_component_editor(self, cat, type_name, idx, comp_idx=None):
        is_edit = comp_idx is not None

        win = tk.Toplevel(self)
        win.title("Edit Component" if is_edit else "Add Component")
        win.resizable(False, False)

        row = 0

        # -----------------------------
        # Name
        # -----------------------------
        ttk.Label(win, text="Name *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        name_entry = ttk.Entry(win, width=40)
        name_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # -----------------------------
        # Component Type
        # -----------------------------
        ttk.Label(win, text="Component Type *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        type_cb = ttk.Combobox(
            win,
            values=list(COMPONENT_TYPES.keys()),
            state="readonly",
            width=37
        )
        type_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # -----------------------------
        # Allowed Actions (Selectable)
        # -----------------------------
        ttk.Label(win, text="Allowed Actions").grid(row=row, column=0, padx=5, sticky="nw")
        action_listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, height=6)
        action_listbox.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        components = inventory[cat][type_name][idx]["components"]

        # -----------------------------
        # States editor (dynamic)
        # -----------------------------
        states_frame = ttk.LabelFrame(win, text="States (state specification)")
        states_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=8, sticky="ew")
        row += 1

        states_widgets = {}

        def clear_states_frame():
            for child in states_frame.winfo_children():
                child.destroy()
            states_widgets.clear()

        def build_states_ui(ctype, existing_states=None):
            clear_states_frame()

            # Predefined enum state specs (e.g. binary control: [0, 1])
            if ctype in COMPONENT_STATE_SPECS:
                spec = COMPONENT_STATE_SPECS[ctype]
                defaults = spec["values"]

                ttk.Label(
                    states_frame,
                    text=f"Default states: {defaults}"
                ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))

                ttk.Label(
                    states_frame,
                    text="Custom labels (optional, comma-separated):"
                ).grid(row=1, column=0, sticky="w", padx=6, pady=(2, 3))

                custom_e = ttk.Entry(states_frame, width=30)
                custom_e.grid(row=1, column=1, sticky="w", padx=6, pady=(2, 3))

                ttk.Label(
                    states_frame,
                    text=f"e.g.  off, on  —  leave blank to keep {defaults}",
                    foreground="gray"
                ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

                # Prefill if editing and custom values were previously saved
                if isinstance(existing_states, dict) and existing_states.get("kind") == "enum":
                    saved_vals = existing_states.get("values", defaults)
                    if saved_vals != defaults:
                        custom_e.insert(0, ", ".join(map(str, saved_vals)))

                states_widgets["binary_custom"] = custom_e
                return

            # Selector control -> enum
            if ctype == "selector control":
                ttk.Label(states_frame, text="Allowed values (comma-separated) *") \
                    .grid(row=0, column=0, sticky="w", padx=5, pady=3)
                e = ttk.Entry(states_frame, width=40)
                e.grid(row=0, column=1, padx=5, pady=3, sticky="w")

                if isinstance(existing_states, dict) and existing_states.get("kind") == "enum":
                    e.insert(0, ",".join(map(str, existing_states.get("values", []))))

                states_widgets["selector_values"] = e
                return

            # Continuous control -> numeric
            if ctype == "continuous control":
                ttk.Label(states_frame, text="Min *") \
                    .grid(row=0, column=0, sticky="w", padx=5, pady=3)
                min_e = ttk.Entry(states_frame, width=14)
                min_e.grid(row=0, column=1, padx=5, pady=3, sticky="w")

                ttk.Label(states_frame, text="Max *") \
                    .grid(row=1, column=0, sticky="w", padx=5, pady=3)
                max_e = ttk.Entry(states_frame, width=14)
                max_e.grid(row=1, column=1, padx=5, pady=3, sticky="w")

                ttk.Label(states_frame, text="Step (optional)") \
                    .grid(row=2, column=0, sticky="w", padx=5, pady=3)
                step_e = ttk.Entry(states_frame, width=14)
                step_e.grid(row=2, column=1, padx=5, pady=3, sticky="w")

                if isinstance(existing_states, dict) and existing_states.get("kind") == "numeric":
                    min_e.insert(0, str(existing_states.get("min", "")))
                    max_e.insert(0, str(existing_states.get("max", "")))
                    if "step" in existing_states:
                        step_e.insert(0, str(existing_states["step"]))

                states_widgets["cont_min"] = min_e
                states_widgets["cont_max"] = max_e
                states_widgets["cont_step"] = step_e
                return

            # Observable -> numeric + unit
            if ctype == "observable":

                mode_var = tk.StringVar(value="numeric")

                # --- symbolic widgets ---
                values_label = ttk.Label(states_frame, text="Values")
                values_entry = ttk.Entry(states_frame, width=30)

                # --- numeric widgets ---
                min_label = ttk.Label(states_frame, text="Min")
                min_e = ttk.Entry(states_frame, width=10)

                max_label = ttk.Label(states_frame, text="Max")
                max_e = ttk.Entry(states_frame, width=10)

                unit_label = ttk.Label(states_frame, text="Unit")
                unit_e = ttk.Entry(states_frame, width=10)

                # -------- toggle logic --------
                def update_obs_mode():

                    if mode_var.get() == "symbolic":

                        min_label.grid_remove()
                        min_e.grid_remove()

                        max_label.grid_remove()
                        max_e.grid_remove()

                        unit_label.grid_remove()
                        unit_e.grid_remove()

                        values_label.grid(row=1, column=0, sticky="w")
                        values_entry.grid(row=1, column=1, columnspan=2, sticky="w")

                    else:

                        values_label.grid_remove()
                        values_entry.grid_remove()

                        min_label.grid(row=1, column=0, sticky="w")
                        min_e.grid(row=1, column=1, sticky="w")

                        max_label.grid(row=2, column=0, sticky="w")
                        max_e.grid(row=2, column=1, sticky="w")

                        unit_label.grid(row=3, column=0, sticky="w")
                        unit_e.grid(row=3, column=1, sticky="w")

                # -------- observable type selector --------
                ttk.Label(states_frame, text="Observable Type").grid(row=0, column=0, sticky="w")

                mode_numeric = ttk.Radiobutton(
                    states_frame,
                    text="Numeric",
                    variable=mode_var,
                    value="numeric",
                    command=update_obs_mode
                )

                mode_symbolic = ttk.Radiobutton(
                    states_frame,
                    text="Symbolic",
                    variable=mode_var,
                    value="symbolic",
                    command=update_obs_mode
                )

                mode_numeric.grid(row=0, column=1, sticky="w")
                mode_symbolic.grid(row=0, column=2, sticky="w")
                
                # -----------------------------
                # Prefill existing observable states
                # -----------------------------
                if isinstance(existing_states, dict):

                    if existing_states.get("kind") == "enum":

                        mode_var.set("symbolic")

                        values = existing_states.get("values", [])
                        values_entry.insert(0, ", ".join(map(str, values)))

                    elif existing_states.get("kind") == "numeric":

                        mode_var.set("numeric")

                        min_e.insert(0, str(existing_states.get("min", "")))
                        max_e.insert(0, str(existing_states.get("max", "")))
                        unit_e.insert(0, str(existing_states.get("unit", "")))

                update_obs_mode()

                states_widgets["obs_mode"] = mode_var
                states_widgets["obs_values"] = values_entry
                states_widgets["obs_min"] = min_e
                states_widgets["obs_max"] = max_e
                states_widgets["obs_unit"] = unit_e
                return

            ttk.Label(states_frame, text="No state editor for this component type.").pack(anchor="w", padx=6, pady=6)

        def update_actions_and_states(existing_states=None):
            # rebuild actions listbox for this type
            action_listbox.delete(0, tk.END)
            ctype = type_cb.get()
            for action in COMPONENT_TYPES.get(ctype, []):
                action_listbox.insert(tk.END, action)

            # rebuild states UI, optionally with existing states
            build_states_ui(ctype, existing_states)

        # bind user changes (no existing_states on user change)
        def on_type_change(event=None):
            update_actions_and_states(existing_states=None)

        type_cb.bind("<<ComboboxSelected>>", on_type_change)

        # -----------------------------
        # Edit Mode Prefill
        # -----------------------------
        if is_edit:
            comp = components[comp_idx]

            # name + type
            name_entry.insert(0, comp.get("name", ""))
            type_cb.set(comp.get("type", ""))

            # build actions+states UI ONCE with existing states
            update_actions_and_states(existing_states=comp.get("states"))

            # preselect actions
            existing_actions = set(comp.get("actions", []))
            allowed_actions = COMPONENT_TYPES.get(comp.get("type", ""), [])
            for i, action in enumerate(allowed_actions):
                if action in existing_actions:
                    action_listbox.select_set(i)

        # -----------------------------
        # Confirm Logic
        # -----------------------------
        def confirm():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Name required.")
                return

            ctype = type_cb.get()
            if not ctype:
                messagebox.showerror("Error", "Component type required.")
                return

            allowed = COMPONENT_TYPES.get(ctype, [])
            selected_indices = action_listbox.curselection()
            selected_actions = [allowed[i] for i in selected_indices]

            comp_data = {
                "name": name,
                "type": ctype,
                "actions": selected_actions
            }

            # -----------------------------
            # Build typed 'states'
            # -----------------------------
    
            # Receptor → dynamic domain (computed later in MDP generation)
            if ctype == "receptor":
                comp_data["states"] = {
                    "kind": "dynamic_receptor"
                }

            elif ctype in COMPONENT_STATE_SPECS:
                spec = copy.deepcopy(COMPONENT_STATE_SPECS[ctype])
                raw_custom = states_widgets.get("binary_custom", None)
                if raw_custom is not None:
                    custom_text = raw_custom.get().strip()
                    if custom_text:
                        custom_vals = [v.strip() for v in custom_text.split(",") if v.strip()]
                        if len(custom_vals) != len(spec["values"]):
                            messagebox.showerror(
                                "Error",
                                f"Binary control requires exactly {len(spec['values'])} custom labels "
                                f"(got {len(custom_vals)})."
                            )
                            return
                        spec["values"] = custom_vals
                comp_data["states"] = spec

            elif ctype == "selector control":
                raw = states_widgets["selector_values"].get().strip()
                values = [v.strip() for v in raw.split(",") if v.strip()]
                if not values:
                    messagebox.showerror("Error", "Selector requires allowed values.")
                    return
                comp_data["states"] = {"kind": "enum", "values": values}

            elif ctype == "continuous control":
                try:
                    mn = float(states_widgets["cont_min"].get().strip())
                    mx = float(states_widgets["cont_max"].get().strip())
                except:
                    messagebox.showerror("Error", "Continuous control requires numeric Min and Max.")
                    return
                if mx < mn:
                    messagebox.showerror("Error", "Max must be >= Min.")
                    return

                spec = {"kind": "numeric", "min": mn, "max": mx}
                step_txt = states_widgets["cont_step"].get().strip()
                if step_txt:
                    try:
                        spec["step"] = float(step_txt)
                    except:
                        messagebox.showerror("Error", "Step must be numeric.")
                        return

                comp_data["states"] = spec

            elif ctype == "observable":

                mode = states_widgets["obs_mode"].get()

                if mode == "symbolic":

                    raw = states_widgets["obs_values"].get().strip()
                    values = [v.strip() for v in raw.split(",") if v.strip()]

                    if not values:
                        messagebox.showerror("Error", "Observable symbolic values required.")
                        return

                    comp_data["states"] = {
                        "kind": "enum",
                        "values": values
                    }

                else:

                    try:
                        mn = float(states_widgets["obs_min"].get())
                        mx = float(states_widgets["obs_max"].get())
                    except:
                        messagebox.showerror("Error", "Numeric observable requires min/max.")
                        return

                    unit = states_widgets["obs_unit"].get().strip()

                    comp_data["states"] = {
                        "kind": "numeric",
                        "min": mn,
                        "max": mx,
                        "unit": unit
                    }

            else:
                comp_data["states"] = None

            # save
            if is_edit:
                components[comp_idx] = comp_data
                focus = ("component", cat, type_name, str(idx), str(comp_idx))
            else:
                components.append(comp_data)
                new_idx = len(components) - 1
                focus = ("component", cat, type_name, str(idx), str(new_idx))

            win.destroy()
            self.after_change(focus_values=focus)

        ttk.Button(win, text="Save" if is_edit else "Add", command=confirm)\
            .grid(row=row, column=1, pady=15, sticky="e")

        self.center_window(win)
        win.grab_set()

    def open_item_editor_clone(self, category, type_name, cloned_item):

        win = tk.Toplevel(self)
        win.title("Clone Item")
        win.resizable(False, False)

        row = 0

        # Category
        ttk.Label(win, text="Category *").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        cat_cb = ttk.Combobox(
            win,
            values=["instrument", "tool", "container", "material"],
            state="readonly",
            width=35
        )
        cat_cb.grid(row=row, column=1, padx=5, pady=5)
        cat_cb.set(category)
        row += 1

        # Type Name
        ttk.Label(win, text="Type name *").grid(row=row, column=0, sticky="w", padx=5)
        type_entry = ttk.Entry(win, width=40)
        type_entry.grid(row=row, column=1, padx=5, pady=5)
        type_entry.insert(0, type_name)
        row += 1

        # Object Actions (if applicable)
        clone_action_entry = None
        if category != "material":
            ttk.Label(win, text="Object Actions").grid(row=row, column=0, padx=5, sticky="w")
            ttk.Label(win, text="(comma-separated)", foreground="gray").grid(row=row, column=1, sticky="w", padx=5)
            row += 1
            clone_action_entry = ttk.Entry(win, width=40)
            clone_action_entry.grid(row=row, column=1, padx=5, pady=5)
            existing = cloned_item.get("object_actions", [])
            clone_action_entry.insert(0, ", ".join(existing))
            row += 1

        # Material field (containers only)
        material_entry = None
        if category == "container":
            ttk.Label(win, text="Material (optional)").grid(row=row, column=0, sticky="w", padx=5)
            material_entry = ttk.Entry(win, width=40)
            material_entry.grid(row=row, column=1, padx=5, pady=5)

            contains = cloned_item.get("contains")
            if contains:
                material_entry.insert(0, contains.get("type_name", ""))

            row += 1

        # Confirm clone
        def confirm():
            new_category = cat_cb.get()
            new_type = type_entry.get().strip()

            if not new_category or not new_type:
                messagebox.showerror("Error", "Category and type name required.")
                return

            inventory[new_category].setdefault(new_type, [])
            new_id = len(inventory[new_category][new_type])

            # Use deep copy again to avoid mutation
            new_item = copy.deepcopy(cloned_item)
            new_item["id"] = new_id

            if new_category != "material" and clone_action_entry is not None:
                raw_actions = clone_action_entry.get().strip()
                new_item["object_actions"] = [a.strip() for a in raw_actions.split(",") if a.strip()]

            if new_category == "container":
                material_name = material_entry.get().strip()
                if material_name:
                    new_item["contains"] = {
                        "entity_type": "material",
                        "type_name": material_name
                    }
                else:
                    new_item.pop("contains", None)

            inventory[new_category][new_type].append(new_item)

            win.destroy()
            self.after_change(
                focus_values=("item", new_category, new_type, str(new_id))
            )

        ttk.Button(win, text="Clone", command=confirm)\
            .grid(row=row, column=1, pady=15, sticky="e")

        self.center_window(win)
        win.grab_set()

    def get_objects_with_state(self, state_name):
        results = []
        for cat in inventory:
            for tname in inventory[cat]:
                for item in inventory[cat][tname]:
                    if state_name in item.get("states", {}):
                        results.append((cat, tname, str(item["id"])))
        return results


    def get_receptor_components(self):
        results = []
        for cat in inventory:
            for tname in inventory[cat]:
                for item in inventory[cat][tname]:
                    for comp in item.get("components", []):
                        if comp.get("type") == "receptor":
                            results.append((
                                cat,
                                tname,
                                str(item["id"]),
                                comp["name"]
                            ))
        return results

    def open_interaction_editor(self, interaction_id=None):
        is_edit = interaction_id is not None
        interaction = interactions[interaction_id] if is_edit else None

        # Map between internal JSON type and human-readable display label
        DISPLAY_TYPE  = {"place": "move object", "transfer": "transfer material"}
        INTERNAL_TYPE = {"move object": "place", "transfer material": "transfer"}
        DEFAULT_ACTION = {"move object": "place", "transfer material": "transfer"}

        win = tk.Toplevel(self)
        win.title("Edit Interaction" if is_edit else "Add Interaction")
        win.resizable(False, False)

        row = 0

        # ─────────────────────────────────────────
        # Step 1 – Interaction Type
        # ─────────────────────────────────────────
        ttk.Label(win, text="Interaction Type *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        type_cb = ttk.Combobox(win, values=["move object", "transfer material"],
                                state="readonly", width=30)
        type_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # ─────────────────────────────────────────
        # SOURCE section
        # ─────────────────────────────────────────
        source_section_label = ttk.Label(win, text="", font=("", 9, "bold"))
        source_section_label.grid(row=row, column=0, columnspan=2, padx=5, pady=(8, 0), sticky="w")
        row += 1

        ttk.Label(win, text="Category *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        source_cat_cb = ttk.Combobox(win, state="readonly", width=30)
        source_cat_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        ttk.Label(win, text="Type *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        source_type_cb = ttk.Combobox(win, state="readonly", width=30)
        source_type_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        ttk.Label(win, text="ID (or *)").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        source_id_cb = ttk.Combobox(win, state="readonly", width=30)
        source_id_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # ─────────────────────────────────────────
        # TARGET section
        # ─────────────────────────────────────────
        target_section_label = ttk.Label(win, text="", font=("", 9, "bold"))
        target_section_label.grid(row=row, column=0, columnspan=2, padx=5, pady=(8, 0), sticky="w")
        row += 1

        ttk.Label(win, text="Category *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        target_cat_cb = ttk.Combobox(win, state="readonly", width=30)
        target_cat_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        ttk.Label(win, text="Type *").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        target_type_cb = ttk.Combobox(win, state="readonly", width=30)
        target_type_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        ttk.Label(win, text="ID (or *)").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        target_id_cb = ttk.Combobox(win, state="readonly", width=30)
        target_id_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # Receptor component — visible only for "move object"
        target_comp_label = ttk.Label(win, text="Receptor Component *")
        target_comp_cb   = ttk.Combobox(win, width=30, state="readonly")
        target_comp_label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        target_comp_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # ─────────────────────────────────────────
        # Action
        # ─────────────────────────────────────────
        action_sep = ttk.Label(win, text="", font=("", 9, "bold"))
        action_sep.grid(row=row, column=0, columnspan=2, padx=5, pady=(8, 0), sticky="w")
        row += 1

        ttk.Label(win, text="Action").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        action_cb = ttk.Combobox(win, state="readonly", width=30)
        action_cb.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # ─────────────────────────────────────────
        # Helper: resolve available actions for any object ref.
        # For instruments: union of all component actions.
        # For everything else: object_actions.
        # ─────────────────────────────────────────
        def get_object_actions(cat, type_name, obj_id_or_star):
            if not cat or not type_name or not obj_id_or_star:
                return []
            if obj_id_or_star == "*":
                items = inventory.get(cat, {}).get(type_name, [])
                item = items[0] if items else None
            else:
                try:
                    item = inventory[cat][type_name][int(obj_id_or_star)]
                except Exception:
                    item = None
            if not item:
                return []
            if cat == "instrument":
                # Full-path labels: "{action} {component name}", deduplicated, order preserved
                seen = set()
                actions = []
                for comp in item.get("components", []):
                    for a in comp.get("actions", []):
                        label = f"{a} {comp['name']}"
                        if label not in seen:
                            actions.append(label)
                            seen.add(label)
                return actions
            return item.get("object_actions", [])

        # ─────────────────────────────────────────
        # Candidate builders
        # ─────────────────────────────────────────
        def material_capable_candidates():
            """
            Objects that can hold material: anything with a 'quantity' state,
            plus every item in the 'container' category (even if quantity state
            wasn't explicitly declared).
            """
            found = set(map(tuple, self.get_objects_with_state("quantity")))
            for tname in inventory.get("container", {}):
                for item in inventory["container"][tname]:
                    found.add(("container", tname, str(item["id"])))
            return sorted(found)

        def src_candidates():
            t = type_cb.get()
            if t == "move object":
                # movable objects: those with a 'location' state
                return self.get_objects_with_state("location")
            if t == "transfer material":
                return material_capable_candidates()
            return []

        def tgt_candidates():
            t = type_cb.get()
            if t == "move object":
                # objects with a receptor component OR that can contain material
                recs = self.get_receptor_components()
                found = {(c, tn, i) for (c, tn, i, _) in recs}
                for (c, tn, i) in material_capable_candidates():
                    found.add((c, tn, i))
                return sorted(found)
            if t == "transfer material":
                return material_capable_candidates()
            return []

        def receptor_names_for_target(cat, tname, obj_id_or_star):
            if not cat or not tname or not obj_id_or_star:
                return []
            if obj_id_or_star == "*":
                items = inventory.get(cat, {}).get(tname, [])
                item = items[0] if items else None
            else:
                try:
                    item = inventory[cat][tname][int(obj_id_or_star)]
                except Exception:
                    item = None
            if not item:
                return []
            return [c["name"] for c in item.get("components", [])
                    if c.get("type") == "receptor"]

        # ─────────────────────────────────────────
        # Action dropdown refresh
        # ─────────────────────────────────────────
        def refresh_action_options():
            t = type_cb.get()
            default = DEFAULT_ACTION.get(t, t)

            src_actions = get_object_actions(
                source_cat_cb.get(), source_type_cb.get(), source_id_cb.get())

            if t == "transfer material":
                tgt_actions = get_object_actions(
                    target_cat_cb.get(), target_type_cb.get(), target_id_cb.get())
            else:
                tgt_actions = []

            # Default action first, then unique custom actions from source (and target for transfer)
            seen = {default}
            options = [default]
            for a in (src_actions + tgt_actions):
                if a not in seen:
                    options.append(a)
                    seen.add(a)

            current = action_cb.get()
            action_cb["values"] = options
            if current not in options:
                action_cb.set(default)

        # ─────────────────────────────────────────
        # UI rebuild helpers
        # ─────────────────────────────────────────
        def clear_source():
            source_cat_cb.set("");  source_cat_cb["values"]  = []
            source_type_cb.set(""); source_type_cb["values"] = []
            source_id_cb.set("");   source_id_cb["values"]   = []

        def clear_target():
            target_cat_cb.set("");  target_cat_cb["values"]  = []
            target_type_cb.set(""); target_type_cb["values"] = []
            target_id_cb.set("");   target_id_cb["values"]   = []
            target_comp_cb.set(""); target_comp_cb["values"] = []

        def refresh_visibility():
            t = type_cb.get()
            if t == "move object":
                source_section_label.config(text="── What to move ──")
                target_section_label.config(text="── Where to place ──")
                action_sep.config(text="── Action ──")
                target_comp_label.grid()
                target_comp_cb.grid()
            elif t == "transfer material":
                source_section_label.config(text="── Source ──")
                target_section_label.config(text="── Target ──")
                action_sep.config(text="── Action ──")
                target_comp_label.grid_remove()
                target_comp_cb.grid_remove()
                target_comp_cb.set("")
                target_comp_cb["values"] = []
            else:
                source_section_label.config(text="")
                target_section_label.config(text="")
                action_sep.config(text="")
                target_comp_label.grid_remove()
                target_comp_cb.grid_remove()

        def rebuild_all(event=None):
            clear_source()
            clear_target()
            refresh_visibility()
            t = type_cb.get()
            if not t:
                action_cb["values"] = []
                action_cb.set("")
                return
            sc = src_candidates()
            source_cat_cb["values"] = sorted(set(c for (c, tn, i) in sc))
            tc = tgt_candidates()
            target_cat_cb["values"] = sorted(set(c for (c, tn, i) in tc))
            # Reset action to the type default
            default = DEFAULT_ACTION.get(t, t)
            action_cb["values"] = [default]
            action_cb.set(default)

        def update_source_types(event=None):
            cat = source_cat_cb.get()
            sc  = src_candidates()
            source_type_cb.set(""); source_id_cb.set("")
            source_type_cb["values"] = sorted(set(tn for (c, tn, i) in sc if c == cat))

        def update_source_ids(event=None):
            cat = source_cat_cb.get()
            tn  = source_type_cb.get()
            sc  = src_candidates()
            ids = [i for (c, tname, i) in sc if c == cat and tname == tn]
            source_id_cb.set("")
            source_id_cb["values"] = ["*"] + sorted(ids, key=lambda x: int(x))

        def on_source_id_selected(event=None):
            refresh_action_options()

        def update_target_types(event=None):
            cat = target_cat_cb.get()
            target_type_cb.set(""); target_id_cb.set("")
            target_comp_cb.set(""); target_comp_cb["values"] = []
            tc = tgt_candidates()
            target_type_cb["values"] = sorted(set(tn for (c, tn, i) in tc if c == cat))

        def update_target_ids(event=None):
            cat = target_cat_cb.get()
            tn  = target_type_cb.get()
            target_id_cb.set(""); target_comp_cb.set(""); target_comp_cb["values"] = []
            tc  = tgt_candidates()
            ids = [i for (c, tname, i) in tc if c == cat and tname == tn]
            target_id_cb["values"] = ["*"] + sorted(ids, key=lambda x: int(x))

        def update_target_components(event=None):
            if type_cb.get() == "move object":
                cat  = target_cat_cb.get()
                tn   = target_type_cb.get()
                oid  = target_id_cb.get()
                comps = receptor_names_for_target(cat, tn, oid)
                target_comp_cb.set(""); target_comp_cb["values"] = comps
                # show receptor selector only if target has receptors
                if comps:
                    target_comp_label.grid()
                    target_comp_cb.grid()
                else:
                    target_comp_label.grid_remove()
                    target_comp_cb.grid_remove()
            refresh_action_options()

        # ─────────────────────────────────────────
        # Bindings
        # ─────────────────────────────────────────
        type_cb.bind("<<ComboboxSelected>>", rebuild_all)

        source_cat_cb.bind("<<ComboboxSelected>>", update_source_types)
        source_type_cb.bind("<<ComboboxSelected>>", update_source_ids)
        source_id_cb.bind("<<ComboboxSelected>>",   on_source_id_selected)

        target_cat_cb.bind("<<ComboboxSelected>>", update_target_types)
        target_type_cb.bind("<<ComboboxSelected>>", update_target_ids)
        target_id_cb.bind("<<ComboboxSelected>>",   update_target_components)

        # ─────────────────────────────────────────
        # Edit mode prefill (order matters)
        # ─────────────────────────────────────────
        if is_edit:
            internal_type = interaction.get("type", "")
            display_type  = DISPLAY_TYPE.get(internal_type, internal_type)
            type_cb.set(display_type)
            rebuild_all()

            src = interaction["source"]
            tgt = interaction["target"]

            source_cat_cb.set(src["category"]);  update_source_types()
            source_type_cb.set(src["type"]);     update_source_ids()
            source_id_cb.set(str(src["id"]))

            target_cat_cb.set(tgt["category"]);  update_target_types()
            target_type_cb.set(tgt["type"]);     update_target_ids()
            target_id_cb.set(str(tgt["id"]))

            if display_type == "move object":
                update_target_components()
                if tgt.get("component"):
                    target_comp_cb.set(tgt["component"])

            # Restore saved action from whichever endpoint carries one.
            # Instrument actions are stored as "comp_norm.verb" → reconstruct UI label "verb comp display".
            # Non-instrument object_actions are stored as plain strings (e.g. "scoop").
            refresh_action_options()
            saved_action = ""
            for ep in (interaction["source"], interaction["target"]):
                ep_action = ep.get("action", "")
                if not ep_action:
                    continue
                if ep.get("category") == "instrument" and "." in ep_action:
                    # reconstruct UI label: "comp_norm.verb" -> "verb comp display"
                    comp_norm, verb = ep_action.rsplit(".", 1)
                    comp_display = comp_norm.replace("_", " ")
                    saved_action = f"{verb} {comp_display}"
                else:
                    # plain object action — use as-is
                    saved_action = ep_action
                break
            if not saved_action:
                saved_action = DEFAULT_ACTION.get(display_type, "")
            if saved_action not in action_cb["values"]:
                opts = list(action_cb["values"]) + [saved_action]
                action_cb["values"] = opts
            action_cb.set(saved_action)

        else:
            type_cb.set("")
            clear_source(); clear_target()
            refresh_visibility()
            action_cb["values"] = []
            action_cb.set("")

        # ─────────────────────────────────────────
        # Confirm
        # ─────────────────────────────────────────
        def confirm():
            display_t      = type_cb.get()
            if not display_t:
                messagebox.showerror("Error", "Select an interaction type.")
                return

            internal_t     = INTERNAL_TYPE[display_t]
            full_action    = action_cb.get()
            default_action = DEFAULT_ACTION.get(display_t, "")

            if not source_cat_cb.get() or not source_type_cb.get() or not source_id_cb.get():
                messagebox.showerror("Error", "Select source (category / type / id).")
                return

            if not target_cat_cb.get() or not target_type_cb.get() or not target_id_cb.get():
                messagebox.showerror("Error", "Select target (category / type / id).")
                return

            if display_t == "move object":
                comps = receptor_names_for_target(
                    target_cat_cb.get(), target_type_cb.get(), target_id_cb.get())
                if comps and not target_comp_cb.get():
                    messagebox.showerror("Error", "Move object requires a target receptor component.")
                    return

            def make_endpoint(cat, type_name, id_str, is_place_target=False):
                """
                Build a source/target dict.
                - id kept as "*" or converted to int.
                - For the move-object target the component comes from the receptor selector.
                - For instrument endpoints whose full_action is one of their component
                  actions, decompose into component + action fields on the endpoint itself.
                """
                obj_id   = id_str  # keep as "*" or original string
                endpoint = {
                    "category":  cat,
                    "type":      type_name,
                    "id":        obj_id,
                    "component": None
                }

                if is_place_target:
                    endpoint["component"] = target_comp_cb.get() or None
                    return endpoint  # component is None for container targets (no receptor)

                if full_action != default_action and full_action in get_object_actions(cat, type_name, id_str):
                    if cat == "instrument":
                        # instrument actions are "verb comp_name" -> store as "comp_norm.verb"
                        parts = full_action.split(" ", 1)
                        if len(parts) == 2:
                            verb, comp_name = parts[0], parts[1]
                            comp_norm = comp_name.replace(" ", "_")
                            endpoint["action"] = f"{comp_norm}.{verb}"
                    else:
                        # non-instrument object_actions are plain strings (e.g. "scoop", "pour")
                        endpoint["action"] = full_action

                return endpoint

            new_interaction = {
                "id":     interaction_id if is_edit else len(interactions),
                "type":   internal_t,
                "source": make_endpoint(
                    source_cat_cb.get(), source_type_cb.get(), source_id_cb.get()
                ),
                "target": make_endpoint(
                    target_cat_cb.get(), target_type_cb.get(), target_id_cb.get(),
                    is_place_target=(display_t == "move object")
                )
            }

            if is_edit:
                interactions[interaction_id] = new_interaction
            else:
                interactions.append(new_interaction)

            save_inventory()
            win.destroy()
            self.refresh_tree()
            self.refresh_json()

        ttk.Button(win, text="Save" if is_edit else "Add", command=confirm)\
            .grid(row=row, column=1, pady=15, sticky="e")

        self.center_window(win)
        win.grab_set()
        
    # -----------------------------
    # TREE OPERATIONS
    # -----------------------------
    def add_component(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "Select an item first.")
            return

        values = self.tree.item(sel[0], "values")

        if not values or values[0] != "item":
            messagebox.showerror("Error", "Select an item (e.g. electronic_scale[0]).")
            return

        if values[1] == "material":
            messagebox.showerror("Error", "Materials cannot have components.")
            return

        self.open_component_editor(values[1], values[2], int(values[3]))

    def edit_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "Select an item or component to edit.")
            return

        values = self.tree.item(sel[0], "values")

        if not values:
            messagebox.showerror("Error", "Select an item or component to edit.")
            return

        if values[0] == "item":
            self.open_item_editor(values[1], values[2], int(values[3]))
        elif values[0] == "component":
            self.open_component_editor(values[1], values[2], int(values[3]), int(values[4]))
        else:
            messagebox.showerror("Error", "Only items or components can be edited.")

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "Select an item or component to delete.")
            return

        values = self.tree.item(sel[0], "values")

        if not values:
            messagebox.showerror("Error", "Only items or components can be deleted.")
            return

        # Confirm deletion
        name = self.tree.item(sel[0], "text")

        if values[0] == "component":
            confirm = messagebox.askyesno(
                "Confirm Delete",
                f"Delete component '{name}'?"
            )
        elif values[0] == "item":
            confirm = messagebox.askyesno(
                "Confirm Delete",
                f"Delete item '{name}' and all its components?"
            )
        elif values[0] == "interaction":
            inter_id = int(values[1])

            confirm = messagebox.askyesno(
                "Confirm Delete",
                f"Delete interaction {inter_id}?"
            )
            if not confirm:
                return

            # --- Check if interactions node is expanded ---
            interactions_open = False
            for node in self.tree.get_children(""):
                if self.tree.item(node, "text") == "interactions":
                    interactions_open = self.tree.item(node, "open")
                    break

            # --- Delete interaction ---
            interactions[:] = [i for i in interactions if i["id"] != inter_id]

            # Reindex
            for idx, inter in enumerate(interactions):
                inter["id"] = idx

            save_inventory()
            self.refresh_tree()

            # --- Restore expanded state ---
            if interactions_open:
                for node in self.tree.get_children(""):
                    if self.tree.item(node, "text") == "interactions":
                        self.tree.item(node, open=True)

            self.refresh_json()
            return
        
        else:
            messagebox.showerror("Error", "Only items or components can be deleted.")
            return

        if not confirm:
            return

        focus_kind = None
        focus_data = None

        if values[0] == "component":
            cat, tname, idx, cidx = values[1], values[2], int(values[3]), int(values[4])
            inventory[cat][tname][idx]["components"].pop(cidx)

            focus_kind = "item"
            focus_data = (cat, tname, values[3])

        elif values[0] == "item":
            cat, tname, idx = values[1], values[2], int(values[3])

            inventory[cat][tname].pop(idx)

            for i, it in enumerate(inventory[cat][tname]):
                it["id"] = i

            if inventory[cat][tname]:
                focus_kind = "type"
                focus_data = (cat, tname)
            else:
                inventory[cat].pop(tname)
                focus_kind = "category"
                focus_data = (cat,)

        save_inventory()
        self.refresh_tree()

        node = None

        if focus_kind == "item":
            node = self.restore_selection(
                ("item", focus_data[0], focus_data[1], focus_data[2])
            )
        elif focus_kind == "type":
            node = self.restore_type_node(focus_data[0], focus_data[1])
        elif focus_kind == "category":
            node = self.restore_category_node(focus_data[0])

        if node:
            self.expand_to_node(node)
            self.tree.selection_set(node)
            self.tree.see(node)

        self.refresh_json()

    # -----------------------------
    # Tree + Focus
    # -----------------------------
    def restore_selection(self, values):
        if not values:
            return None
        for node in self.tree.get_children(""):
            found = self._restore_recursive(node, values)
            if found:
                return found
        return None

    def _restore_recursive(self, node, values):
        node_values = self.tree.item(node, "values")
        if node_values and node_values == values:
            return node
        for child in self.tree.get_children(node):
            found = self._restore_recursive(child, values)
            if found:
                return found
        return None

    def expand_to_node(self, node):
        while node:
            self.tree.item(node, open=True)
            node = self.tree.parent(node)

    def after_change(self, focus_values=None):
        save_inventory()
        self.refresh_tree()
        node = self.restore_selection(focus_values)
        if node:
            self.expand_to_node(node)
            self.tree.selection_set(node)
            self.tree.see(node)
        self.refresh_json()

    def restore_type_node(self, cat, tname):
        for node in self.tree.get_children(""):
            if self.tree.item(node, "text") == cat:
                for child in self.tree.get_children(node):
                    if self.tree.item(child, "text") == tname:
                        return child
        return None

    def restore_category_node(self, cat):
        for node in self.tree.get_children(""):
            if self.tree.item(node, "text") == cat:
                return node
        return None

    # -----------------------------
    # Refresh
    # -----------------------------
    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())

        # -----------------------------
        # OBJECTS
        # -----------------------------
        objects_root = self.tree.insert("", "end", text="objects")

        for cat, types in inventory.items():
            cat_node = self.tree.insert(objects_root, "end", text=cat)

            for tname, items in types.items():
                type_node = self.tree.insert(cat_node, "end", text=tname)

                for item in items:
                    label = f"{tname}[{item['id']}]"

                    # ---------------------------------
                    # DERIVE FLAGS FROM STATES
                    # ---------------------------------
                    states = item.get("states", {})

                    flags = []
                    if "location" in states:
                        flags.append("movable")
                    if "quantity" in states:
                        flags.append("container")

                    if flags:
                        label += " [" + ", ".join(flags) + "]"

                    # Object actions summary
                    if cat != "material" and item.get("object_actions"):
                        label += f" ⚙{len(item['object_actions'])}"

                    # Container material summary
                    if cat == "container" and item.get("contains"):
                        label += f" → {item['contains']['type_name']}"

                    item_node = self.tree.insert(
                        type_node,
                        "end",
                        text=label,
                        values=("item", cat, tname, str(item["id"]))
                    )

                    # -----------------------------
                    # COMPONENTS
                    # -----------------------------
                    if cat != "material":
                        for i, comp in enumerate(item.get("components", [])):
                            comp_label = f"{comp['name']} ({comp['type']})"

                            # --- STATE SUMMARY ---
                            states = comp.get("states")

                            if isinstance(states, dict):

                                kind = states.get("kind")

                                # Static enum
                                if kind == "enum":
                                    n = len(states.get("values", []))
                                    comp_label += f" Σ{n}"

                                # Numeric
                                elif kind == "numeric":
                                    mn = states.get("min")
                                    mx = states.get("max")
                                    unit = states.get("unit", "")
                                    if unit:
                                        comp_label += f" [{mn}–{mx} {unit}]"
                                    else:
                                        comp_label += f" [{mn}–{mx}]"

                                # Dynamic receptor
                                elif kind == "dynamic_receptor":
                                    domain = self.get_receptor_domain(cat, tname, item["id"], comp["name"])
                                    comp_label += f" ⟨{', '.join(domain)}⟩"

                            # --- ACTION SUMMARY ---
                            if comp.get("actions"):
                                comp_label += f" ⚙{len(comp['actions'])}"

                            self.tree.insert(
                                item_node,
                                "end",
                                text=comp_label,
                                values=("component", cat, tname, str(item["id"]), str(i))
                            )

        # -----------------------------
        # INTERACTIONS
        # -----------------------------
        interactions_root = self.tree.insert("", "end", text="interactions")

        for inter in interactions:
            src = inter["source"]
            tgt = inter["target"]

            # Derive action label from whichever endpoint carries one,
            # falling back to the bare interaction type.
            action_label = inter.get("type", "")
            for ep in (src, tgt):
                ep_action = ep.get("action", "")
                if ep_action:
                    action_label = ep_action
                    break

            def fmt_ep(e):
                return f"{e['category']}:{e['type']}[{e['id']}]"

            label = f"{fmt_ep(src)} -- {action_label} -> {fmt_ep(tgt)}"

            self.tree.insert(
                interactions_root,
                "end",
                text=label,
                values=("interaction", str(inter["id"]))
            )

    def refresh_json(self):
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(world, indent=2))
        self.current_right_panel_filename = "inventory.json"

    def center_window(self, win):
        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    def on_close(self):
        save_inventory()
        self.destroy()

    def clone_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "Select an item to clone.")
            return

        values = self.tree.item(sel[0], "values")

        if not values or values[0] != "item":
            messagebox.showerror("Error", "Only items can be cloned.")
            return

        cat, type_name, idx = values[1], values[2], int(values[3])

        original_item = inventory[cat][type_name][idx]

        # Deep copy entire structure
        cloned_item = copy.deepcopy(original_item)

        # Remove id (will be re-assigned)
        cloned_item.pop("id", None)

        # Open editor in clone mode
        self.open_item_editor_clone(cat, type_name, cloned_item)

    def generate_object_mdp_template(self, cat, type_name, obj_id):

        global inventory, interactions

        def norm(x):
            return x.replace(" ", "_").lower() if isinstance(x, str) else x

        def interaction_action(inter, is_source):
            """
            Return the action name for this interaction:
            - Short name when the owning endpoint IS the current object
              e.g. "draw_trigger_button.press" or "scoop"
            - Fully qualified when the owning endpoint is a DIFFERENT object
              e.g. "instrument:electronic_pipette[0].draw_trigger_button.press"
              or   "tool:spoon[*].scoop"
            - Falls back to the bare interaction type if no action is found.
            """
            for ep in (inter["source"], inter["target"]):
                ep_action = ep.get("action", "")
                if not ep_action:
                    continue
                owner_is_self = (
                    ep["category"] == cat and
                    ep["type"] == type_name and
                    ep.get("id") in (str(obj_id), "*")
                )
                if owner_is_self:
                    return ep_action  # short name (dotted or plain)
                else:
                    prefix = f"{ep['category']}:{norm(ep['type'])}[{ep['id']}]"
                    return f"{prefix}.{ep_action}"  # fully qualified
            return inter.get("type", "transfer")

        obj = inventory[cat][type_name][obj_id]

        template = {
            "object": f"{cat}:{norm(type_name)}[{obj_id}]",
            "states": [],
            "actions": []
        }

        # -------------------------------------------------
        # 1️⃣ STATES (components + object states)
        # -------------------------------------------------
        for comp in obj.get("components", []):
            states = comp.get("states")

            if not states:
                continue

            # 🔥 HANDLE dynamic_receptor
            if states["kind"] == "dynamic_receptor":
                values = ["empty"]

                for inter in interactions:
                    if inter["type"] != "place":
                        continue

                    tgt = inter["target"]

                    if (
                        tgt["category"] == cat and
                        tgt["type"] == type_name and
                        tgt.get("component") == comp["name"]
                    ):
                        src = inter["source"]
                        obj_label = f"{src['category']}:{norm(src['type'])}"
                        values.append(obj_label)

                template["states"].append({
                    "name": norm(comp["name"]),
                    "kind": "enum",
                    "values": sorted(set(values))
                })

            # 🔹 NORMAL STATES
            else:
                template["states"].append({
                    "name": norm(comp["name"]),
                    "kind": states["kind"],
                    **{k: v for k, v in states.items() if k != "kind"}
                })

        for state_name, state_info in obj.get("states", {}).items():
            # resolve material domain for the main object itself
            if state_info.get("domain") == "materials":
                possible_materials = set()
                for inter2 in interactions:
                    if inter2["type"] != "transfer":
                        continue
                    tgt2 = inter2["target"]
                    if not (tgt2["category"] == cat and tgt2["type"] == type_name and
                            tgt2.get("id") in (str(obj_id), "*")):
                        continue
                    src2 = inter2["source"]
                    if src2["category"] == "container":
                        src2_id = int(src2["id"]) if src2["id"] != "*" else 0
                        src2_obj = inventory[src2["category"]][src2["type"]][src2_id]
                        contains = src2_obj.get("contains")
                        if contains and contains.get("entity_type") == "material":
                            possible_materials.add(f"material:{norm(contains['type_name'])}")
                    elif src2["category"] in ("tool", "instrument"):
                        for inter3 in interactions:
                            if inter3["type"] != "transfer":
                                continue
                            if not (inter3["target"]["category"] == src2["category"] and
                                    inter3["target"]["type"] == src2["type"]):
                                continue
                            src3 = inter3["source"]
                            if src3["category"] == "container":
                                src3_id = int(src3["id"]) if src3["id"] != "*" else 0
                                src3_obj = inventory[src3["category"]][src3["type"]][src3_id]
                                contains = src3_obj.get("contains")
                                if contains and contains.get("entity_type") == "material":
                                    possible_materials.add(f"material:{norm(contains['type_name'])}")
                if possible_materials:
                    template["states"].append({
                        "name": norm(state_name),
                        "kind": "enum",
                        "values": ["none"] + sorted(possible_materials)
                    })
                # if no materials found, skip — domain filter will drop it anyway
                continue

            template["states"].append({
                "name": norm(state_name),
                "kind": state_info["kind"],
                **{k: v for k, v in state_info.items() if k != "kind"}
            })

        # -------------------------------------------------
        # 1b️⃣ STATES OF INTERACTED OBJECTS
        # -------------------------------------------------
        seen_interacted = set()
        
        def add_interacted_states(other_cat, other_type, other_id):

            def get_material_state(instance, label):
                mat_state_info = instance.get("states", {}).get("material")
                if not mat_state_info or mat_state_info.get("domain") != "materials":
                    return None

                possible_materials = set()

                def trace_material_from_source(src):
                    if src["category"] == "container":
                        src_id = int(src["id"]) if src["id"] != "*" else 0
                        src_obj = inventory[src["category"]][src["type"]][src_id]
                        contains = src_obj.get("contains")
                        if contains and contains.get("entity_type") == "material":
                            possible_materials.add(f"material:{norm(contains['type_name'])}")
                    elif src["category"] in ("tool", "instrument"):
                        for inter3 in interactions:
                            if inter3["type"] != "transfer":
                                continue
                            if (inter3["target"]["category"] == src["category"] and
                                    inter3["target"]["type"] == src["type"]):
                                src3 = inter3["source"]
                                if src3["category"] == "container":
                                    src3_id = int(src3["id"]) if src3["id"] != "*" else 0
                                    src3_obj = inventory[src3["category"]][src3["type"]][src3_id]
                                    contains = src3_obj.get("contains")
                                    if contains and contains.get("entity_type") == "material":
                                        possible_materials.add(f"material:{norm(contains['type_name'])}")

                for inter2 in interactions:
                    if inter2["type"] != "transfer":
                        continue

                    src2 = inter2["source"]
                    tgt2 = inter2["target"]

                    current_involved = (
                        (src2["category"] == cat and src2["type"] == type_name and
                         src2.get("id") in (str(obj_id), "*")) or
                        (tgt2["category"] == cat and tgt2["type"] == type_name and
                         tgt2.get("id") in (str(obj_id), "*"))
                    )
                    if not current_involved:
                        continue

                    # CASE A: interacted object directly involved in this interaction
                    other_involved = (
                        (src2["category"] == other_cat and src2["type"] == other_type) or
                        (tgt2["category"] == other_cat and tgt2["type"] == other_type)
                    )
                    if other_involved:
                        trace_material_from_source(src2)
                        continue

                    # CASE B: 1-hop mediated — current → mediator → interacted object
                    # inter2 is current → mediator, look for mediator → interacted object
                    mediator_cat = tgt2["category"]
                    mediator_type = tgt2["type"]

                    for inter3 in interactions:
                        if inter3["type"] != "transfer":
                            continue
                        if not (inter3["source"]["category"] == mediator_cat and
                                inter3["source"]["type"] == mediator_type):
                            continue
                        if not (inter3["target"]["category"] == other_cat and
                                inter3["target"]["type"] == other_type):
                            continue
                        # mediator → interacted object found — trace material from current object
                        trace_material_from_source(src2)

                if not possible_materials:
                    return None

                return {
                    "name": f"{label}.material",
                    "kind": "enum",
                    "values": ["material:none"] + sorted(possible_materials)
                }

            def get_location_domain(other_cat, other_type, other_id):
                loc_state_info = inventory[other_cat].get(other_type, [{}])[0].get("states", {}).get("location")
                if not loc_state_info or loc_state_info.get("domain") != "locations":
                    return None

                values = []
                for inter2 in interactions:
                    if inter2["type"] != "place":
                        continue

                    src2 = inter2["source"]
                    tgt2 = inter2["target"]

                    # interacted object must be the source of this place interaction
                    if src2["category"] != other_cat or src2["type"] != other_type:
                        continue

                    # current object must be the target
                    if not (tgt2["category"] == cat and tgt2["type"] == type_name and
                            tgt2.get("id") in (str(obj_id), "*")):
                        continue

                    receptor = norm(tgt2["component"]) if tgt2.get("component") else None
                    loc = f"{cat}:{norm(type_name)}[{obj_id}]"
                    if receptor:
                        loc += f":{receptor}"
                    values.append(loc)

                return ["bench"] + sorted(set(values)) if values else None

            if other_id == "*":
                instances = inventory[other_cat].get(other_type, [])
                if not instances:
                    return
                instance = instances[0]
                label = f"{other_cat}:{norm(other_type)}"

                if label in seen_interacted:
                    return
                seen_interacted.add(label)

                for comp in instance.get("components", []):
                    comp_states = comp.get("states")
                    if not comp_states:
                        continue
                    state_name = f"{label}.{norm(comp['name'])}"
                    if comp_states["kind"] == "dynamic_receptor":
                        values = self.get_receptor_domain(
                            other_cat, other_type, instance["id"], comp["name"]
                        )
                        template["states"].append({"name": state_name, "kind": "enum", "values": values})
                    else:
                        template["states"].append({
                            "name": state_name,
                            "kind": comp_states["kind"],
                            **{k: v for k, v in comp_states.items() if k != "kind"}
                        })

                for sname, sinfo in instance.get("states", {}).items():
                    # skip material domain state — resolved separately by get_material_state
                    if sinfo.get("domain") == "materials":
                        continue
                    # resolve location domain
                    if sinfo.get("domain") == "locations":
                        loc_values = get_location_domain(other_cat, other_type, other_id)
                        if loc_values:
                            template["states"].append({
                                "name": f"{label}.{norm(sname)}",
                                "kind": "enum",
                                "values": loc_values
                            })
                        # if no locations found, drop it (will be caught by domain filter anyway)
                        continue
                    
                    template["states"].append({
                        "name": f"{label}.{norm(sname)}",
                        "kind": sinfo["kind"],
                        **{k: v for k, v in sinfo.items() if k != "kind"}
                    })

                mat_state = get_material_state(instance, label)
                if mat_state:
                    template["states"].append(mat_state)

            else:
                instance = inventory[other_cat][other_type][int(other_id)]
                label = f"{other_cat}:{norm(other_type)}[{other_id}]"

                if label in seen_interacted:
                    return
                seen_interacted.add(label)

                for comp in instance.get("components", []):
                    comp_states = comp.get("states")
                    if not comp_states:
                        continue
                    state_name = f"{label}.{norm(comp['name'])}"
                    if comp_states["kind"] == "dynamic_receptor":
                        values = self.get_receptor_domain(
                            other_cat, other_type, int(other_id), comp["name"]
                        )
                        template["states"].append({"name": state_name, "kind": "enum", "values": values})
                    else:
                        template["states"].append({
                            "name": state_name,
                            "kind": comp_states["kind"],
                            **{k: v for k, v in comp_states.items() if k != "kind"}
                        })

                for sname, sinfo in instance.get("states", {}).items():
                    # skip material domain state — resolved separately by get_material_state
                    if sinfo.get("domain") == "materials":
                        continue
                    # resolve location domain
                    if sinfo.get("domain") == "locations":
                        loc_values = get_location_domain(other_cat, other_type, other_id)
                        if loc_values:
                            template["states"].append({
                                "name": f"{label}.{norm(sname)}",
                                "kind": "enum",
                                "values": loc_values
                            })
                        # if no locations found, drop it (will be caught by domain filter anyway)
                        continue
                    template["states"].append({
                        "name": f"{label}.{norm(sname)}",
                        "kind": sinfo["kind"],
                        **{k: v for k, v in sinfo.items() if k != "kind"}
                    })

                mat_state = get_material_state(instance, label)
                if mat_state:
                    template["states"].append(mat_state)

        # direct interactions
        for inter in interactions:
            src = inter["source"]
            tgt = inter["target"]

            is_source = (
                src["category"] == cat and
                src["type"] == type_name and
                src.get("id") in (str(obj_id), "*")
            )
            is_target = (
                tgt["category"] == cat and
                tgt["type"] == type_name and
                tgt.get("id") in (str(obj_id), "*")
            )

            if not (is_source or is_target):
                continue

            other = tgt if is_source else src
            add_interacted_states(other["category"], other["type"], other["id"])

            # 1-hop mediated: if current object is source of a transfer,
            # also collect the final target of any onward transfer from the mediator
            if is_source and inter["type"] == "transfer":
                mediator_cat = tgt["category"]
                mediator_type = tgt["type"]

                for inter2 in interactions:
                    if inter2["type"] != "transfer":
                        continue
                    if (
                        inter2["source"]["category"] == mediator_cat and
                        inter2["source"]["type"] == mediator_type
                    ):
                        final = inter2["target"]
                        add_interacted_states(final["category"], final["type"], final["id"])
        
        # remove unresolved domain states, quantity states, and material states for containers
        def should_keep(s):
            if "domain" in s:
                return False
            name = s.get("name", "")
            if name == "quantity" or name.endswith(".quantity"):
                return False
            # drop material state if main object is a container
            if cat == "container" and name == "material":
                return False
            # drop material state for interacted container objects
            if name.startswith("container:") and name.endswith(".material"):
                return False
            return True

        template["states"] = [s for s in template["states"] if should_keep(s)]
        
        # -------------------------------------------------
        # 2️⃣ CONTROL ACTIONS
        # -------------------------------------------------
        for comp in obj.get("components", []):
            for action in comp.get("actions", []):
                action_entry = {
                    "name": f"{norm(comp['name'])}.{action}",
                    "type": "control",
                    "parameters": {}
                }

                if comp.get("states") and comp["states"]["kind"] == "enum":
                    action_entry["parameters"]["value"] = comp["states"]["values"]

                template["actions"].append(action_entry)

        # -------------------------------------------------
        # 3️⃣ INTERACTION ACTIONS
        # -------------------------------------------------
        for inter in interactions:

            src = inter["source"]
            tgt = inter["target"]

            is_source = (
                src["category"] == cat and
                src["type"] == type_name and
                src.get("id") in (str(obj_id), "*")
            )

            is_target = (
                tgt["category"] == cat and
                tgt["type"] == type_name and
                tgt.get("id") in (str(obj_id), "*")
            )

            if not (is_source or is_target):
                continue

            # -------------------------------------------------
            # 🔹 PLACE (object-based)
            # -------------------------------------------------
            if inter["type"] == "place":
                if not is_target:
                    continue

                if src["id"] == "*":
                    obj_label = f"{src['category']}:{norm(src['type'])}"
                else:
                    obj_label = f"{src['category']}:{norm(src['type'])}[{src['id']}]"

                action_entry = {
                    "name": interaction_action(inter, is_source=False),
                    "type": "interaction",
                    "parameters": {
                        "object": obj_label,
                        "target": norm(tgt["component"]) if tgt.get("component") else None
                    }
                }

                if action_entry not in template["actions"]:
                    template["actions"].append(action_entry)

                continue

            # -------------------------------------------------
            # 🔹 TRANSFER (material-based)
            # -------------------------------------------------
            if inter["type"] == "transfer":

                # ---------------------------------------------
                # CASE A: current object is TARGET
                # ---------------------------------------------
                if is_target:
                    material_labels = []

                    # source is container
                    if src["category"] == "container":
                        if src["id"] == "*":
                            source_obj = inventory[src["category"]][src["type"]][0]
                        else:
                            source_obj = inventory[src["category"]][src["type"]][int(src["id"])]

                        contains = source_obj.get("contains")
                        if contains and contains.get("entity_type") == "material":
                            material_labels.append(
                                norm(contains['type_name'])
                            )

                    # source is tool/instrument
                    elif src["category"] in ["tool", "instrument"]:
                        for inter2 in interactions:
                            if inter2["type"] != "transfer":
                                continue

                            if (
                                inter2["target"]["category"] == src["category"] and
                                inter2["target"]["type"] == src["type"]
                            ):
                                src2 = inter2["source"]

                                if src2["category"] == "container":
                                    if src2["id"] == "*":
                                        source_obj = inventory[src2["category"]][src2["type"]][0]
                                    else:
                                        source_obj = inventory[src2["category"]][src2["type"]][int(src2["id"])]

                                    contains = source_obj.get("contains")
                                    if contains and contains.get("entity_type") == "material":
                                        material_labels.append(
                                            f"{src['category']}:{norm(src['type'])}:{norm(contains['type_name'])}"
                                        )

                    for mat in set(material_labels):
                        params = {
                            "material": mat,
                            "target_object": f"{cat}:{norm(type_name)}[{obj_id}]"
                        }

                        action_entry = {
                            "name": interaction_action(inter, is_source=False),
                            "type": "interaction",
                            "parameters": params
                        }

                        if action_entry not in template["actions"]:
                            template["actions"].append(action_entry)

                # ---------------------------------------------
                # CASE B: current object is SOURCE
                # ---------------------------------------------
                else:
                    if tgt["id"] == "*":
                        other_label = f"{tgt['category']}:{norm(tgt['type'])}"
                    else:
                        other_label = f"{tgt['category']}:{norm(tgt['type'])}[{tgt['id']}]"

                    material_labels = []

                    # object is container
                    contains = obj.get("contains")
                    if contains and contains.get("entity_type") == "material":
                        base_material = norm(contains['type_name'])
                        material_labels.append(base_material)

                    # object is tool/instrument
                    for inter2 in interactions:
                        if inter2["type"] != "transfer":
                            continue

                        if (
                            inter2["target"]["category"] == cat and
                            inter2["target"]["type"] == type_name
                        ):
                            src2 = inter2["source"]

                            if src2["category"] == "container":
                                if src2["id"] == "*":
                                    source_obj = inventory[src2["category"]][src2["type"]][0]
                                else:
                                    source_obj = inventory[src2["category"]][src2["type"]][int(src2["id"])]

                                contains2 = source_obj.get("contains")
                                if contains2 and contains2.get("entity_type") == "material":
                                    material_labels.append(
                                        f"{cat}:{norm(type_name)}:{norm(contains2['type_name'])}"
                                    )

                    # direct actions for the current interaction
                    for mat in set(material_labels):
                        params = {
                            "material": mat,
                            "target_object": other_label
                        }

                        action_entry = {
                            "name": interaction_action(inter, is_source=True),
                            "type": "interaction",
                            "parameters": params
                        }

                        if action_entry not in template["actions"]:
                            template["actions"].append(action_entry)

                    # -------------------------------------------------
                    # 1-hop mediated downstream transfers from containers
                    # container -> mediator  and  mediator -> final_target
                    # -------------------------------------------------
                    contains = obj.get("contains")
                    if contains and contains.get("entity_type") == "material":
                        material_name = norm(contains["type_name"])

                        for inter_mid in interactions:
                            if inter_mid["type"] != "transfer":
                                continue

                            mid_src = inter_mid["source"]
                            mid_tgt = inter_mid["target"]

                            # current container -> mediator
                            if not (
                                mid_src["category"] == cat and
                                mid_src["type"] == type_name and
                                mid_src.get("id") in (str(obj_id), "*")
                            ):
                                continue

                            mediator_cat = mid_tgt["category"]
                            mediator_type = mid_tgt["type"]

                            mediated_material = f"{mediator_cat}:{norm(mediator_type)}:{material_name}"

                            for inter_next in interactions:
                                if inter_next["type"] != "transfer":
                                    continue

                                next_src = inter_next["source"]
                                next_tgt = inter_next["target"]

                                # mediator -> final target
                                if not (
                                    next_src["category"] == mediator_cat and
                                    next_src["type"] == mediator_type
                                ):
                                    continue

                                if next_tgt["id"] == "*":
                                    final_target_label = f"{next_tgt['category']}:{norm(next_tgt['type'])}"
                                else:
                                    final_target_label = f"{next_tgt['category']}:{norm(next_tgt['type'])}[{next_tgt['id']}]"

                                action_entry = {
                                    "name": interaction_action(inter_next, is_source=False),
                                    "type": "interaction",
                                    "parameters": {
                                        "material": mediated_material,
                                        "target_object": final_target_label
                                    }
                                }

                                if action_entry not in template["actions"]:
                                    template["actions"].append(action_entry)

        # -------------------------------------------------
        # 4️⃣ DEDUPLICATION: drop control entries whose short name matches
        #    the trailing component.action of an interaction entry.
        # -------------------------------------------------
        interaction_short_names = set()
        for a in template["actions"]:
            if a["type"] == "interaction":
                # extract trailing "comp.verb" from fully-qualified or bare name
                parts = a["name"].split(".")
                if len(parts) >= 2:
                    interaction_short_names.add(f"{parts[-2]}.{parts[-1]}")

        template["actions"] = [
            a for a in template["actions"]
            if not (a["type"] == "control" and a["name"] in interaction_short_names)
        ]

        return template

    def generate_selected_mdp(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "Select an object first.")
            return

        values = self.tree.item(sel[0], "values")

        if not values or values[0] != "item":
            messagebox.showerror("Error", "Select an object (not component).")
            return

        cat, type_name, obj_id = values[1], values[2], int(values[3])

        template = self.generate_object_mdp_template(cat, type_name, obj_id)

        # Display result in JSON preview
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(template, indent=2))

        # ---- Set suggested filename for saving ----
        safe_type = type_name.replace(" ", "_")
        self.current_right_panel_filename = f"{cat}_{safe_type}_{obj_id}_mdp_template.json"

    def generate_all_mdp(self):
        # -----------------------------------------------
        # Build list of all objects
        # -----------------------------------------------
        all_objects = []
        for cat in inventory:
            for type_name in inventory[cat]:
                for item in inventory[cat][type_name]:
                    all_objects.append((cat, type_name, item["id"]))

        if not all_objects:
            messagebox.showerror("Error", "No objects in inventory.")
            return

        # -----------------------------------------------
        # Selection dialog
        # -----------------------------------------------
        sel_win = tk.Toplevel(self)
        sel_win.title("Select Objects for MDP Generation")
        sel_win.resizable(False, False)

        ttk.Label(sel_win, text="Select objects to generate MDP templates for:").pack(
            anchor="w", padx=10, pady=(10, 2)
        )

        # Scrollable listbox
        frame = ttk.Frame(sel_win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(
            frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
            width=50,
            height=20
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for cat, type_name, obj_id in all_objects:
            listbox.insert(tk.END, f"{cat}:{type_name.replace(' ', '_')}[{obj_id}]")

        # Select all by default
        listbox.select_set(0, tk.END)

        # Select all / deselect all buttons
        btn_frame = ttk.Frame(sel_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Button(
            btn_frame, text="Select All",
            command=lambda: listbox.select_set(0, tk.END)
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame, text="Deselect All",
            command=lambda: listbox.selection_clear(0, tk.END)
        ).pack(side=tk.LEFT, padx=2)

        # -----------------------------------------------
        # Confirm
        # -----------------------------------------------
        def confirm():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showerror("Error", "No objects selected.")
                return

            selected_objects = [all_objects[i] for i in selected_indices]

            # Ask for output directory
            out_dir = filedialog.askdirectory(title="Select Output Directory")
            if not out_dir:
                return

            sel_win.destroy()

            # Generate and save
            errors = []
            for cat, type_name, obj_id in selected_objects:
                try:
                    template = self.generate_object_mdp_template(cat, type_name, obj_id)
                    safe_type = type_name.replace(" ", "_")
                    filename = f"{cat}_{safe_type}_{obj_id}_mdp_template.json"
                    filepath = os.path.join(out_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(template, f, indent=2)
                except Exception as e:
                    errors.append(f"{cat}:{type_name}[{obj_id}]: {e}")

            if errors:
                messagebox.showwarning(
                    "Completed with errors",
                    f"Generated with {len(errors)} error(s):\n" + "\n".join(errors)
                )
            else:
                messagebox.showinfo(
                    "Done",
                    f"Generated {len(selected_objects)} MDP template(s) in:\n{out_dir}"
                )

        ttk.Button(sel_win, text="Generate", command=confirm).pack(pady=10)

        self.center_window(sel_win)
        sel_win.grab_set()

    def show_inventory_json(self):
        self.refresh_json()

    def save_right_panel_json(self):

        content = self.json_text.get("1.0", tk.END).strip()

        if not content:
            messagebox.showerror("Error", "Nothing to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            initialfile=self.current_right_panel_filename,
            filetypes=[("JSON files", "*.json")]
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            messagebox.showinfo("Saved", f"File saved:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -----------------------------
    # Graph View
    # -----------------------------
    def visualize_graph(self):
        G = nx.DiGraph()

        for cat, types in inventory.items():
            G.add_node(cat, kind="category", level=0)

            for tname, items in types.items():
                G.add_node(tname, kind="type", level=1)
                G.add_edge(cat, tname)

                for item in items:
                    item_node = f"{tname}[{item['id']}]"
                    G.add_node(item_node, kind="item", level=2)
                    G.add_edge(tname, item_node)

                    for comp in item.get("components", []):
                        comp_node = f"{item_node}:{comp['name']}"
                        G.add_node(comp_node, kind="component", level=3)
                        G.add_edge(item_node, comp_node)

                        for action in comp.get("actions", []):
                            action_node = f"{comp_node}:{action}"
                            G.add_node(action_node, kind="action", level=4)
                            G.add_edge(comp_node, action_node)

        # -------------------------
        # Hierarchical Layout (Auto-Scaled)
        # -------------------------
        levels = {}
        for node, data in G.nodes(data=True):
            lvl = data.get("level", 0)
            levels.setdefault(lvl, []).append(node)

        num_levels = len(levels)
        max_width = max(len(nodes) for nodes in levels.values())

        # Auto-scale gaps
        horizontal_gap = max(1.5, 14 / max_width)
        vertical_gap = max(1.5, 10 / num_levels)

        pos = {}

        for level, nodes in levels.items():
            y = -level * vertical_gap
            x_start = -(len(nodes) - 1) * horizontal_gap / 2

            for i, node in enumerate(nodes):
                x = x_start + i * horizontal_gap
                pos[node] = (x, y)

        # -------------------------
        # Color mapping
        # -------------------------
        color_map = {
            "category": "#8ecae6",
            "type": "#f4a261",
            "item": "#90be6d",
            "component": "#ffafcc",
            "action": "#ffe066"
        }

        node_colors = [color_map[G.nodes[n]["kind"]] for n in G.nodes]

        plt.figure(figsize=(16, 10))

        # Draw straight vertical edges
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            plt.plot([x1, x2], [y1, y2], color="gray", linewidth=1)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # -------------------------
        # Legend
        # -------------------------
        import matplotlib.patches as mpatches

        legend_elements = [
            mpatches.Patch(color=color_map["category"], label="Category"),
            mpatches.Patch(color=color_map["type"], label="Type"),
            mpatches.Patch(color=color_map["item"], label="Item"),
            mpatches.Patch(color=color_map["component"], label="Component"),
            mpatches.Patch(color=color_map["action"], label="Action"),
        ]

        plt.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=10,
            frameon=True
        )

        plt.title("Lab Inventory Hierarchical Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = LabInventoryApp()
    app.mainloop()

