import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt

INVENTORY_FILE = "inventory.json"

COMPONENT_TYPES = {
    "binary control": ["toggle", "open", "close"],
    "momentary control": ["press"],
    "continuous control": ["set", "increase", "decrease"],
    "selector control": ["select"],
    "receptor": ["place_object", "remove_object"],
    "actuator": ["activate", "deactivate"],
    "observable": ["read"]
}

OBJECT_ACTIONS = [
    "pick_up",
    "place",
    "move",
    "rotate",
    "attach",
    "detach"
]

# -----------------------------
# Persistence
# -----------------------------
def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "instrument": {},
        "container": {},
        "tool": {},
        "material": {}
    }

def save_inventory(inv):
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)

inventory = load_inventory()

# Ensure fields exist
for cat in inventory:
    for tname in inventory[cat]:
        for item in inventory[cat][tname]:

            if cat != "material":
                item.setdefault("components", [])
                item.setdefault("object_actions", [])

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
        ttk.Button(control_frame, text="Visualize Graph", command=self.visualize_graph).pack(fill=tk.X, pady=5)

        self.tree = ttk.Treeview(left_frame, show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<Double-1>", self.on_double_click)

        # -----------------------------
        # RIGHT PANEL (JSON Preview)
        # -----------------------------
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)  # smaller weight → narrower

        ttk.Label(right_frame, text="Inventory JSON").pack(anchor=tk.W)

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
        ttk.Label(win, text="Object Actions").grid(row=row, column=0, padx=5)
        action_listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, height=6)
        action_listbox.grid(row=row, column=1, padx=5, pady=5)

        for action in OBJECT_ACTIONS:
            action_listbox.insert(tk.END, action)

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

            existing_actions = inventory[cat][type_name][idx].get("object_actions", [])
            for i, action in enumerate(OBJECT_ACTIONS):
                if action in existing_actions:
                    action_listbox.select_set(i)

            if cat == "container":
                existing_material = inventory[cat][type_name][idx].get("material")
                if existing_material:
                    material_entry.insert(0, existing_material.get("name", ""))

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

            selected_indices = action_listbox.curselection()
            selected_object_actions = [OBJECT_ACTIONS[i] for i in selected_indices]

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
                        "object_actions": selected_object_actions
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
        # Save Button (Always Visible)
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
        # Update actions when type changes
        # -----------------------------
        def update_actions(event=None):
            action_listbox.delete(0, tk.END)
            ctype = type_cb.get()
            for action in COMPONENT_TYPES.get(ctype, []):
                action_listbox.insert(tk.END, action)

        type_cb.bind("<<ComboboxSelected>>", update_actions)

        # -----------------------------
        # Edit Mode Prefill
        # -----------------------------
        if is_edit:
            comp = components[comp_idx]
            name_entry.insert(0, comp["name"])
            type_cb.set(comp["type"])
            update_actions()

            existing = comp.get("actions", [])
            for i, action in enumerate(COMPONENT_TYPES.get(comp["type"], [])):
                if action in existing:
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

            if is_edit:
                components[comp_idx] = comp_data
                focus = ("component", cat, type_name, str(idx), str(comp_idx))
            else:
                components.append(comp_data)
                new_idx = len(components) - 1
                focus = ("component", cat, type_name, str(idx), str(new_idx))

            win.destroy()
            self.after_change(focus_values=focus)

        # -----------------------------
        # Save Button (Always Visible)
        # -----------------------------
        ttk.Button(
            win,
            text="Save" if is_edit else "Add",
            command=confirm
        ).grid(row=row, column=1, pady=15, sticky="e")

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

        save_inventory(inventory)
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
        save_inventory(inventory)
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
        for cat, types in inventory.items():
            cat_node = self.tree.insert("", "end", text=cat)
            for tname, items in types.items():
                type_node = self.tree.insert(cat_node, "end", text=tname)
                for item in items:
                    label = f"{tname}[{item['id']}]"

                    # Object-level actions (only for non-material)
                    if cat != "material" and item.get("object_actions"):
                        label += f" ⚙{len(item['object_actions'])}"

                    # Contained material (containers only)
                    if cat == "container":
                        contained = item.get("contains")
                        if contained:
                            label += f" → {contained.get('type_name')}"

                    item_node = self.tree.insert(
                        type_node,
                        "end",
                        text=label,
                        values=("item", cat, tname, str(item["id"]))
                    )

                    # Components (only for non-material)
                    if cat != "material":
                        for i, comp in enumerate(item.get("components", [])):
                            comp_label = f"{comp['name']} ({comp['type']})"
                            if comp.get("actions"):
                                comp_label += f" ⚙{len(comp['actions'])}"
                            self.tree.insert(
                                item_node,
                                "end",
                                text=comp_label,
                                values=("component", cat, tname, str(item["id"]), str(i))
                            )

    def refresh_json(self):
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(inventory, indent=2))

    def center_window(self, win):
        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    def on_close(self):
        save_inventory(inventory)
        self.destroy()

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

