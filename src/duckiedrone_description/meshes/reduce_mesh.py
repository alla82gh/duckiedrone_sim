import bpy

# Import OBJ
bpy.ops.import_scene.obj(filepath="duckiedrone.obj")

# Select the object
obj = bpy.context.selected_objects[0]
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Add decimate modifier
modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
modifier.ratio = 0.1  # Reduce to 10% of polygons (adjust as needed)

# Apply modifier
bpy.ops.object.modifier_apply(modifier="Decimate")

# Export reduced OBJ
bpy.ops.export_scene.obj(filepath="duckiedrone_reduced.obj", use_selection=True)