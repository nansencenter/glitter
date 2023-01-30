FROM continuumio/anaconda3

# Image metadata
LABEL autor='artem.moiseev@nersc.no'
LABEL description='Environment for processing drone videos \
                  to derive wave spectrum'
LABEL github='nansencenter/glitter'
LABEL version='0.1'

# Install ffmpeg to convert videos
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y ffmpeg

# Install python dependencies
RUN conda update --all && \
    conda install jupyterlab numpy matplotlib && \
    conda install -c conda-forge netcdf4 && \
    conda install -c anaconda ipykernel pip
    
EXPOSE 8888