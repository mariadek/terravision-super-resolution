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
        libgdal-jp2openjpeg \
        pyyaml \
        h5py \
        boto3 \
        httpx \
    && conda clean -afy

ENV PATH="/opt/conda/envs/arosics/bin:$PATH"

RUN pip install --no-cache-dir "arosics>=1.3.0"

WORKDIR /app

# Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

WORKDIR /app

CMD ["python", "src/main.py"]



