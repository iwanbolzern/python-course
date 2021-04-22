#!/bin/bash
cd zp3_v1
VER=$(poetry version -s)
poetry version "${VER}.${BUILD_NR}"
poetry publish --build -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD}
cd ..
cd zp3_v2
VER=$(poetry version -s)
poetry version "${VER}.${BUILD_NR}"
poetry publish --build -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD}
cd ..
cd zp1
VER=$(poetry version -s)
poetry version "${VER}.${BUILD_NR}"
poetry publish --build -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD}
cd ..
cd zp2
VER=$(poetry version -s)
poetry version "${VER}.${BUILD_NR}"
poetry publish --build -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD}
cd ..