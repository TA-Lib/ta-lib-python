# Dockerfile for TA-Lib. To build:
#
#    docker build --rm -t talib .
#
# To run:
#
#    docker run --rm -it talib bash
#

ARG PYTHON_VERSION="3.12"

FROM python:$PYTHON_VERSION as builder

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
RUN python -m pip install --no-cache-dir /opt/ta-lib-python/wheels/*.whl \
    && python -c 'import numpy, talib; close = numpy.random.random(100); output = talib.SMA(close); print(output)'
