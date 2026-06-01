import numpy as np
import matplotlib.pyplot as plt
import rioxarray  # activates the .rio accessor

import numpy as np
import matplotlib.pyplot as plt


def plot_sentinel2_vs_tessera_pca(
    image,
    embeddings,
    vector_data=None,
    image_crs="EPSG:32632",
    boundary_color="white",
    figsize=(14, 7),
    max_size=1024,
    pca_sample_size=50000,
    random_seed=42,
    title="Sentinel-2 RGB vs. TESSERA PCA",
):
    """
    Plot a normal Sentinel-2 RGB image next to a PCA visualization
    of TESSERA embeddings.

    Left panel:
        Sentinel-2 RGB image using B04, B03, B02.

    Right panel:
        TESSERA embeddings compressed to 3 PCA components and shown as RGB.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    if embeddings.rio.crs is None:
        embeddings = embeddings.rio.write_crs(image_crs)

    # Match TESSERA to Sentinel-2 grid if needed.
    # This keeps the comparison spatially aligned.
    embeddings = embeddings.rio.reproject_match(image)

    # Downsample only for plotting speed.
    y_size = image.sizes["y"]
    x_size = image.sizes["x"]

    stride = max(1, int(np.ceil(max(y_size, x_size) / max_size)))

    image_plot = image.isel(
        y=slice(None, None, stride),
        x=slice(None, None, stride),
    )

    emb_plot = embeddings.isel(
        y=slice(None, None, stride),
        x=slice(None, None, stride),
    )

    extent = [
        float(image_plot.x.min()),
        float(image_plot.x.max()),
        float(image_plot.y.min()),
        float(image_plot.y.max()),
    ]

    # -----------------------------
    # Prepare Sentinel-2 RGB image
    # -----------------------------
    red = image_plot.sel(band="B04")
    green = image_plot.sel(band="B03")
    blue = image_plot.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    # -----------------------------
    # Prepare TESSERA PCA RGB image
    # -----------------------------
    arr = emb_plot.values.astype("float32")
    n_bands, n_y, n_x = arr.shape

    pixel_table = arr.reshape(n_bands, n_y * n_x).T

    valid_pixels = np.isfinite(pixel_table).all(axis=1)
    valid_data = pixel_table[valid_pixels]

    if valid_data.shape[0] == 0:
        raise ValueError("No valid TESSERA embedding pixels found for PCA plotting.")

    rng = np.random.default_rng(random_seed)

    n_sample = min(pca_sample_size, valid_data.shape[0])
    sample_idx = rng.choice(
        valid_data.shape[0],
        size=n_sample,
        replace=False,
    )

    pca = PCA(n_components=3, random_state=random_seed)
    pca.fit(valid_data[sample_idx])

    pca_values = pca.transform(valid_data)

    pca_rgb_flat = np.full(
        (n_y * n_x, 3),
        fill_value=np.nan,
        dtype="float32",
    )

    pca_rgb_flat[valid_pixels] = pca_values
    pca_rgb = pca_rgb_flat.reshape(n_y, n_x, 3)

    # Robust stretch each PCA component to 0-1.
    for i in range(3):
        component = pca_rgb[:, :, i]
        valid_component = component[np.isfinite(component)]

        p2 = np.nanpercentile(valid_component, 2)
        p98 = np.nanpercentile(valid_component, 98)

        pca_rgb[:, :, i] = (component - p2) / (p98 - p2 + 1e-6)

    pca_rgb = np.clip(pca_rgb, 0, 1)

    explained = pca.explained_variance_ratio_ * 100

    # -----------------------------
    # Plot side by side
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(
        rgb_plot,
        extent=extent,
        origin="upper",
    )
    axes[0].set_title("Sentinel-2 RGB")

    axes[1].imshow(
        pca_rgb,
        extent=extent,
        origin="upper",
    )
    axes[1].set_title(
        "TESSERA PCA RGB\n"
        f"PC1 {explained[0]:.1f}%, "
        f"PC2 {explained[1]:.1f}%, "
        f"PC3 {explained[2]:.1f}%"
    )

    if vector_data is not None:
        vector_plot = vector_data.to_crs(image_plot.rio.crs)

        for ax in axes:
            vector_plot.boundary.plot(
                ax=ax,
                edgecolor=boundary_color,
                linewidth=0.8,
            )

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()

    return fig, axes

def plot_sentinel2_cube_rgb(
    cube,
    time_index=0,
    vector_data=None,
    image_crs="EPSG:32632",
    title=None,
    boundary_color="yellow",
    figsize=(8, 8),
):
    """
    Plot one RGB time slice from a Sentinel-2 xarray data cube.

    The cube is expected to have dimensions: time, band, y, x.
    """

    # Select one date so that the time series becomes one normal image.
    image = cube.isel(time=time_index).compute()

    # If the cube does not carry CRS metadata, assign the expected Sentinel-2 CRS.
    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    red = image.sel(band="B04")
    green = image.sel(band="B03")
    blue = image.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    # Sentinel-2 values may be stored as scaled reflectance.
    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    # Stretch only for visualization, not for analysis.
    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)

    rgb_plot = np.transpose(rgb, (1, 2, 0))

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        rgb_plot,
        extent=[
            float(image.x.min()),
            float(image.x.max()),
            float(image.y.min()),
            float(image.y.max()),
        ],
        origin="upper",
    )

    if vector_data is not None:
        vector_plot = vector_data.to_crs(image.rio.crs)
        vector_plot.boundary.plot(
            ax=ax,
            edgecolor=boundary_color,
            linewidth=1,
        )

    if title is None:
        date = str(image.time.values)[:10]
        title = f"Sentinel-2 RGB observation, {date}"

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return image


from shapely.geometry import box







def plot_sentinel2_image_rgb(
    image,
    vector_data=None,
    comparison_image=None,
    title="Median Sentinel-2 image",
    comparison_title="First Sentinel-2 observation",
    boundary_color="yellow",
    image_crs="EPSG:32632",
    figsize=(14, 7),
):
    """
    Plot an RGB composite from a Sentinel-2 image.

    If comparison_image is provided, the function plots the comparison image
    and the main image side by side.
    """

    def prepare_image(img):
        # Ensure the image has CRS metadata so vector boundaries can be aligned.
        if img.rio.crs is None:
            img = img.rio.write_crs(image_crs)
        return img

    def make_rgb(img):
        # Use visible bands to create a natural-colour RGB image.
        red = img.sel(band="B04")
        green = img.sel(band="B03")
        blue = img.sel(band="B02")

        rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

        # Sentinel-2 values may be stored as scaled reflectance.
        if np.nanmax(rgb) > 1.5:
            rgb = rgb / 10000.0

        # Stretch only for display, not for analysis.
        p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
        p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

        rgb = (rgb - p2) / (p98 - p2 + 1e-6)
        rgb = np.clip(rgb, 0, 1)

        return np.transpose(rgb, (1, 2, 0))

    def get_extent(img):
        # Convert x/y coordinates to the extent expected by matplotlib.
        return [
            float(img.x.min()),
            float(img.x.max()),
            float(img.y.min()),
            float(img.y.max()),
        ]

    def clipped_boundaries(img, gdf):
        # Clip district boundaries to the image footprint so they do not extend
        # beyond the raster edge in the plot.
        if gdf is None:
            return None

        vector_plot = gdf.to_crs(img.rio.crs)

        xmin, xmax = float(img.x.min()), float(img.x.max())
        ymin, ymax = float(img.y.min()), float(img.y.max())

        image_box = box(xmin, ymin, xmax, ymax)
        return vector_plot.clip(image_box)

    image = prepare_image(image)

    if comparison_image is not None:
        comparison_image = prepare_image(comparison_image)

        images = [comparison_image, image]
        titles = [comparison_title, title]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, img, panel_title in zip(axes, images, titles):
            ax.imshow(
                make_rgb(img),
                extent=get_extent(img),
                origin="upper",
            )

            boundaries = clipped_boundaries(img, vector_data)
            if boundaries is not None:
                boundaries.boundary.plot(
                    ax=ax,
                    edgecolor=boundary_color,
                    linewidth=1,
                )

            ax.set_title(panel_title)
            ax.set_axis_off()

    else:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(
            make_rgb(image),
            extent=get_extent(image),
            origin="upper",
        )

        boundaries = clipped_boundaries(image, vector_data)
        if boundaries is not None:
            boundaries.boundary.plot(
                ax=ax,
                edgecolor=boundary_color,
                linewidth=1,
            )

        ax.set_title(title)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return image



from shapely.geometry import box
from matplotlib.patches import Patch


def plot_classified_suburbs_on_sentinel2(
    image,
    suburbs,
    class_column="class",
    name_column="BEZIR_NAME",
    title="Classified Göttingen districts over Sentinel-2 median image",
    image_crs="EPSG:32632",
    figsize=(10, 10),
):
    """
    Plot classified suburb polygons on top of a Sentinel-2 RGB image.

    The Sentinel-2 image is expected to have dimensions: band, y, x.
    The suburb GeoDataFrame must contain a class column, for example:
    urban, forest, fields.
    """

    # Ensure the raster has CRS metadata so the vector data can be aligned.
    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    # Create a natural-colour RGB background from Sentinel-2 bands.
    red = image.sel(band="B04")
    green = image.sel(band="B03")
    blue = image.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    # Stretch only for display.
    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    extent = [
        float(image.x.min()),
        float(image.x.max()),
        float(image.y.min()),
        float(image.y.max()),
    ]

    # Reproject and clip the suburb polygons to the Sentinel-2 image extent.
    suburbs_plot = suburbs.to_crs(image.rio.crs)

    xmin, xmax = float(image.x.min()), float(image.x.max())
    ymin, ymax = float(image.y.min()), float(image.y.max())
    image_box = box(xmin, ymin, xmax, ymax)

    suburbs_plot = suburbs_plot.clip(image_box)

    class_colors = {
        "urban": "#d7191c",
        "forest": "#1a9641",
        "fields": "#fdae61",
    }

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        rgb_plot,
        extent=extent,
        origin="upper",
    )

    # Draw filled class polygons.
    suburbs_plot.plot(
        ax=ax,
        column=class_column,
        categorical=True,
        color=suburbs_plot[class_column].map(class_colors),
        alpha=0.35,
        edgecolor="black",
        linewidth=1,
    )

    # Draw a stronger boundary overlay.
    suburbs_plot.boundary.plot(
        ax=ax,
        edgecolor="white",
        linewidth=1.2,
    )

    suburbs_plot.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.4,
    )

    # Add district name and class text.
    for _, row in suburbs_plot.iterrows():
        point = row.geometry.representative_point()

        label = f"{row[name_column]}\n{row[class_column]}"

        ax.text(
            point.x,
            point.y,
            label,
            fontsize=7,
            ha="center",
            va="center",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.65,
            },
        )

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in class_colors.items()
    ]

    ax.legend(
        handles=legend_handles,
        title="Class",
        loc="lower left",
        frameon=True,
    )

    ax.set_title(title)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return suburbs_plot


def plot_class_spectral_profiles_plot(
    spectra_df,
    class_column="class",
    value_column="mean_reflectance",
    std_column="std_reflectance",
    wavelength_column="wavelength_nm",
    label_column="name",
    class_order=("urban", "forest", "fields"),
    title="Average Sentinel-2 spectral profiles by class",
    figsize=(16, 6),
    interpolation_kind="linear",
    n_interp_points=500,
    show_uncertainty=True,
):
    """
    Plot class spectral profiles in two ways:
    1. Evenly spaced Sentinel-2 bands with spectral-region labels.
    2. Wavelength-scaled plot with interpolated lines.

    The first subplot is easier to read as a band-by-band comparison.
    The second subplot shows the physical spacing between wavelengths.
    """

    from scipy.interpolate import interp1d

    required_columns = {
        class_column,
        value_column,
        wavelength_column,
        label_column,
    }

    missing = required_columns - set(spectra_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = spectra_df.copy()

    band_labels = (
        df[[wavelength_column, label_column]]
        .drop_duplicates()
        .sort_values(wavelength_column)
        .reset_index(drop=True)
    )

    x_regular = np.arange(len(band_labels))
    wavelengths = band_labels[wavelength_column].values

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax_regular, ax_wave = axes

    for class_name in class_order:
        class_df = (
            df[df[class_column] == class_name]
            .sort_values(wavelength_column)
            .reset_index(drop=True)
        )

        if class_df.empty:
            continue

        y = class_df[value_column].values
        x_wave = class_df[wavelength_column].values

        # Subplot 1: evenly spaced bands.
        line = ax_regular.plot(
            x_regular,
            y,
            marker="o",
            linewidth=2,
            label=class_name.capitalize(),
        )

        color = line[0].get_color()

        if show_uncertainty and std_column in class_df.columns:
            y_std = class_df[std_column].fillna(0).values
            ax_regular.fill_between(
                x_regular,
                y - y_std,
                y + y_std,
                alpha=0.15,
                color=color,
            )

        # Subplot 2: physical wavelength axis with interpolation.
        # Linear interpolation is safest with Sentinel-2's uneven band spacing,
        # especially across the large gap between NIR and SWIR.
        interpolator = interp1d(
            x_wave,
            y,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value="extrapolate",
        )

        x_interp = np.linspace(x_wave.min(), x_wave.max(), n_interp_points)
        y_interp = interpolator(x_interp)

        ax_wave.plot(
            x_interp,
            y_interp,
            linewidth=2,
            color=color,
            label=class_name.capitalize(),
        )

        ax_wave.scatter(
            x_wave,
            y,
            color=color,
            s=35,
            zorder=3,
        )

    # Left subplot: readable categorical band spacing.
    ax_regular.set_xticks(x_regular)
    ax_regular.set_xticklabels(
        band_labels[label_column],
        rotation=45,
        ha="right",
    )
    ax_regular.set_xlabel("Sentinel-2 spectral region")
    ax_regular.set_ylabel("Mean reflectance")
    ax_regular.set_title("Bands spaced evenly")
    ax_regular.grid(True, linestyle="--", alpha=0.4)
    ax_regular.legend(title="Class")

    # Right subplot: real wavelength spacing.
    ax_wave.set_xticks(wavelengths)
    ax_wave.set_xticklabels(
        [f"{int(w)} nm" for w in wavelengths],
        rotation=45,
        ha="right",
    )
    ax_wave.set_xlabel("Wavelength")
    ax_wave.set_ylabel("Mean reflectance")
    ax_wave.set_title("Interpolated by wavelength")
    ax_wave.grid(True, linestyle="--", alpha=0.4)
    ax_wave.legend(title="Class")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_rgb_ndvi_evi(
    image,
    ndvi,
    evi,
    vector_data=None,
    title="Sentinel-2 mean image with vegetation indices",
    image_crs="EPSG:32632",
    boundary_color="white",
    figsize=(18, 6),
):
    """
    Plot the Sentinel-2 RGB image together with NDVI and EVI maps.

    This links the vegetation indices back to the original satellite image.
    """

    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    if ndvi.rio.crs is None:
        ndvi = ndvi.rio.write_crs(image_crs)

    if evi.rio.crs is None:
        evi = evi.rio.write_crs(image_crs)

    red = image.sel(band="B04")
    green = image.sel(band="B03")
    blue = image.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    extent = [
        float(image.x.min()),
        float(image.x.max()),
        float(image.y.min()),
        float(image.y.max()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(rgb_plot, extent=extent, origin="upper")
    axes[0].set_title("Sentinel-2 RGB")

    im1 = axes[1].imshow(
        ndvi.values,
        extent=extent,
        origin="upper",
        cmap="YlGn",
        vmin=-0.2,
        vmax=0.9,
    )
    axes[1].set_title("NDVI")

    im2 = axes[2].imshow(
        evi.values,
        extent=extent,
        origin="upper",
        cmap="YlGn",
        vmin=-0.2,
        vmax=0.9,
    )
    axes[2].set_title("EVI")

    if vector_data is not None:
        vector_plot = vector_data.to_crs(image.rio.crs)

        for ax in axes:
            vector_plot.boundary.plot(
                ax=ax,
                edgecolor=boundary_color,
                linewidth=0.8,
            )

    for ax in axes:
        ax.set_axis_off()

    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="NDVI")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="EVI")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_class_index_comparison(
    class_indices_df,
    class_column="class",
    class_order=("urban", "forest", "fields"),
    figsize=(10, 5),
    title="Vegetation indices by class",
):
    """
    Compare average NDVI and EVI values across land-cover classes.

    The values should already be aggregated by district first, so that each
    district contributes equally to the class average.
    """

    required_columns = {
        class_column,
        "mean_ndvi",
        "std_ndvi",
        "mean_evi",
        "std_evi",
    }

    missing = required_columns - set(class_indices_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = (
        class_indices_df
        .set_index(class_column)
        .loc[list(class_order)]
        .reset_index()
    )

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - width / 2,
        df["mean_ndvi"],
        width,
        yerr=df["std_ndvi"],
        capsize=4,
        label="NDVI",
    )

    ax.bar(
        x + width / 2,
        df["mean_evi"],
        width,
        yerr=df["std_evi"],
        capsize=4,
        label="EVI",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df[class_column].str.capitalize())
    ax.set_ylabel("Index value")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig, ax



from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def plot_threshold_class_map(
    image,
    class_map,
    vector_data=None,
    title="Threshold-based land-cover map",
    image_crs="EPSG:32632",
    figsize=(14, 7),
):
    """
    Plot a threshold-based class map next to the Sentinel-2 RGB image.

    class_map values:
    0 = urban
    1 = fields
    2 = forest
    """

    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    if class_map.rio.crs is None:
        class_map = class_map.rio.write_crs(image_crs)

    # RGB background
    red = image.sel(band="B04")
    green = image.sel(band="B03")
    blue = image.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    extent = [
        float(image.x.min()),
        float(image.x.max()),
        float(image.y.min()),
        float(image.y.max()),
    ]

    class_colors = {
        0: "#d7191c",  # urban
        1: "#fdae61",  # fields
        2: "#1a9641",  # forest
    }

    class_labels = {
        0: "Urban",
        1: "Fields",
        2: "Forest",
    }

    cmap = ListedColormap([class_colors[0], class_colors[1], class_colors[2]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(rgb_plot, extent=extent, origin="upper")
    axes[0].set_title("Sentinel-2 RGB")

    axes[1].imshow(
        class_map.values,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        alpha=0.85,
    )
    axes[1].set_title(title)

    if vector_data is not None:
        vector_plot = vector_data.to_crs(image.rio.crs)

        for ax in axes:
            vector_plot.boundary.plot(
                ax=ax,
                edgecolor="white",
                linewidth=0.8,
            )

    legend_handles = [
        Patch(facecolor=class_colors[value], edgecolor="black", label=label)
        for value, label in class_labels.items()
    ]

    axes[1].legend(
        handles=legend_handles,
        title="Class",
        loc="lower left",
        frameon=True,
    )

    for ax in axes:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_training_samples(
    image,
    training_gdf,
    vector_data=None,
    class_column="class",
    title="Training samples over Sentinel-2 RGB image",
    image_crs="EPSG:32632",
    figsize=(10, 10),
    sample_size=4,
):
    """
    Plot sampled training pixels on top of a Sentinel-2 RGB image.
    """

    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    if training_gdf.crs != image.rio.crs:
        training_gdf = training_gdf.to_crs(image.rio.crs)

    red = image.sel(band="B04")
    green = image.sel(band="B03")
    blue = image.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    extent = [
        float(image.x.min()),
        float(image.x.max()),
        float(image.y.min()),
        float(image.y.max()),
    ]

    class_colors = {
        "urban": "#d7191c",
        "fields": "#fdae61",
        "forest": "#1a9641",
    }

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        rgb_plot,
        extent=extent,
        origin="upper",
    )

    if vector_data is not None:
        vector_plot = vector_data.to_crs(image.rio.crs)
        vector_plot.boundary.plot(
            ax=ax,
            edgecolor="white",
            linewidth=0.8,
        )

    for class_name, color in class_colors.items():
        subset = training_gdf[training_gdf[class_column] == class_name]

        if subset.empty:
            continue

        subset.plot(
            ax=ax,
            color=color,
            markersize=sample_size,
            label=class_name.capitalize(),
            alpha=0.6,

        )

    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(title="Training class", loc="lower left")

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_order=("urban", "fields", "forest"),
    title="Confusion matrix",
    figsize=(6, 5),
    normalize=True,
):
    """
    Plot a confusion matrix for the land-cover classifier.

    If normalize=True, each row sums to 1. This makes it easier to see
    which classes are confused, independent of class size.
    """

    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(class_order))

    if normalize:
        cm_plot = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        value_format = ".2f"
        colorbar_label = "Proportion of true class"
    else:
        cm_plot = cm
        value_format = "d"
        colorbar_label = "Number of pixels"

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_plot)

    ax.set_xticks(np.arange(len(class_order)))
    ax.set_yticks(np.arange(len(class_order)))

    ax.set_xticklabels([c.capitalize() for c in class_order], rotation=45, ha="right")
    ax.set_yticklabels([c.capitalize() for c in class_order])

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    for i in range(len(class_order)):
        for j in range(len(class_order)):
            value = cm_plot[i, j]
            ax.text(
                j,
                i,
                format(value, value_format),
                ha="center",
                va="center",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.show()

    return fig, ax, cm



from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def plot_prediction_map(
    image,
    prediction_da,
    vector_data=None,
    title="Random Forest land-cover prediction",
    image_crs="EPSG:32632",
    figsize=(14, 7),
):
    """
    Plot the Random Forest prediction map next to the Sentinel-2 RGB image.

    prediction_da values:
    0 = urban
    1 = fields
    2 = forest
    """

    if image.rio.crs is None:
        image = image.rio.write_crs(image_crs)

    if prediction_da.rio.crs is None:
        prediction_da = prediction_da.rio.write_crs(image_crs)

    red = image.sel(band="B04")
    green = image.sel(band="B03")
    blue = image.sel(band="B02")

    rgb = np.stack([red.values, green.values, blue.values]).astype("float32")

    if np.nanmax(rgb) > 1.5:
        rgb = rgb / 10000.0

    p2 = np.nanpercentile(rgb, 2, axis=(1, 2), keepdims=True)
    p98 = np.nanpercentile(rgb, 98, axis=(1, 2), keepdims=True)

    rgb = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    extent = [
        float(image.x.min()),
        float(image.x.max()),
        float(image.y.min()),
        float(image.y.max()),
    ]

    class_colors = {
        0: "#d7191c",  # urban
        1: "#fdae61",  # fields
        2: "#1a9641",  # forest
    }

    class_labels = {
        0: "Urban",
        1: "Fields",
        2: "Forest",
    }

    cmap = ListedColormap([class_colors[0], class_colors[1], class_colors[2]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(rgb_plot, extent=extent, origin="upper")
    axes[0].set_title("Sentinel-2 RGB")

    axes[1].imshow(
        prediction_da.values,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
    )
    axes[1].set_title(title)

    if vector_data is not None:
        vector_plot = vector_data.to_crs(image.rio.crs)

        for ax in axes:
            vector_plot.boundary.plot(
                ax=ax,
                edgecolor="white",
                linewidth=0.8,
            )

    legend_handles = [
        Patch(facecolor=class_colors[value], edgecolor="black", label=label)
        for value, label in class_labels.items()
    ]

    axes[1].legend(
        handles=legend_handles,
        title="Predicted class",
        loc="lower left",
        frameon=True,
    )

    for ax in axes:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return fig, axes

