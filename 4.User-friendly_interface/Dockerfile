FROM tensorflow/tensorflow:2.9.1
MAINTAINER nathalie

# Install ubuntu packages
RUN apt update \
 && apt-get install -y git \
 && apt-get install -y unzip \
 && apt-get install -y  wget \
 && apt-get install -y  curl \
 && apt-get install -y  time

# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install cmake --upgrade
RUN pip3 install scanpy pandas==1.5.3 numpy==1.24.1 openpyxl ipywidgets seaborn matplotlib scikit-learn==1.2.0 rdkit

# Download dependent files for scDrug
#RUN git clone https://github.com/ailabstw/scDrug.git /scDrug
RUN git clone https://github.com/yysun0116/scDrug2.0.git /scDrug

# CaDRReS-Sc
RUN git clone https://github.com/CSB5/CaDRReS-Sc.git /opt/CaDRReS-Sc
RUN sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()/g' /opt/CaDRReS-Sc/cadrres_sc/model.py
RUN sed -i 's/import tensorflow\.python\.util\.deprecation as deprecation/from tensorflow.python.util import deprecation/g' /opt/CaDRReS-Sc/cadrres_sc/model.py


# Load shell script
COPY run.sh /app/
WORKDIR /app

CMD ["sh", "run.sh"]
