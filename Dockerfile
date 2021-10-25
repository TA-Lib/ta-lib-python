# Dockerfile for TA-Lib. To build:
#
#    docker build --rm -t talib .
#
# To run:
#
#    docker run --rm -it talib bash
#

ARG PYTHON_VERSION="3.7"

FROM python:$PYTHON_VERSION as builder

ENV TA_PREFIX="/opt/ta-lib-core"
ENV TA_LIBRARY_PATH="$TA_PREFIX/lib" \
    TA_INCLUDE_PATH="$TA_PREFIX/include"

WORKDIR /src/ta-lib-core
RUN apt-get update && apt-get install -y \
        gfortran \
        libfreetype6-dev \
        libhdf5-dev \
        liblapack-dev \
        libopenblas-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    | tar xvz --strip-components 1 \
    && ./configure --prefix="$TA_PREFIX" \
    && make \
    && make install

WORKDIR /src/ta-lib-python
COPY . .
RUN python -m pip install -e . \
    && python -c 'import numpy, talib; close = numpy.random.random(100); output = talib.SMA(close); print(output)' \
    && python -m pip wheel --wheel-dir wheels .

ARG RUN_TESTS="1"
RUN if [ "$RUN_TESTS" -ne "0" ]; then \
        python -m pip install -r requirements_test.txt \
        && pytest . ; \
    else \
        echo "Skipping tests\n" ; \
    fi

# Build final image.
FROM python:$PYTHON_VERSION-slim
COPY --from=builder /src/ta-lib-python/wheels /opt/ta-lib-python/wheels
COPY --from=builder /opt/ta-lib-core /opt/ta-lib-core
RUN python -m pip install --no-cache-dir /opt/ta-lib-python/wheels/*.whl \
    && python -c 'import numpy, talib; close = numpy.random.random(100); output = talib.SMA(close); print(output)'
