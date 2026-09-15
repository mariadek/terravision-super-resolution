FROM condaforge/miniforge3:latest

RUN conda create -y -n arosics \
        -c conda-forge \
        python=3.11 \
        pip \
        cartopy \
        gdal \
        geopandas \
        numpy \
        scipy \
        shapely \
        pyproj \
        rasterio \
        scikit-image \
        netcdf4 \
        libgdal-jp2openjpeg \
    && conda clean -afy

ENV PATH="/opt/conda/envs/arosics/bin:$PATH"

WORKDIR /app

# Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

WORKDIR /app

CMD ["python", "src/main.py"]


