"""Flat fallback colors on Newton's display model only; never modify USD/camera materials."""
# Existing scene diffuse colors, used when Newton cannot display the textures.
PALETTE={'table':(.12,.45,.22),'robot':(.08,.08,.08),'cube':(.85,.70,.35),'box':(.55,.38,.20)}

def apply_newton_colors(viewer):
    model=viewer.model
    if model is None or model.shape_color is None:return {}
    colors=model.shape_color.numpy()
    counts={name:0 for name in PALETTE}
    for i,label in enumerate(model.shape_label):
        label=label.lower()
        # The tabletop collider itself is named "box". Match scene ancestors
        # before primitive names, so its shape name cannot override Table.
        for name,rgb in PALETTE.items():
            if '/'+name+'/' in label or label.endswith('/'+name) or label.startswith(name):
                colors[i]=rgb;counts[name]+=1;break
    model.shape_color.assign(colors)
    return counts
