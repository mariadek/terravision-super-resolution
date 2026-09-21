FROM condaforge/miniforge3:latest

RUN conda create -y -n arosics \
        -c conda-forge \
        python=3.11 \
        pip=26.2.1 \
        cartopy=0.25.0 \
        gdal=3.13.1 \
        geopandas=1.1.4 \
        numpy=2.4.6 \
        scipy=1.17.1 \
        shapely=2.1.2 \
        pyproj=3.7.2 \
        rasterio=1.4.4 \
        scikit-image=0.26.0 \
        libgdal-jp2openjpeg=3.13.1 \
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



