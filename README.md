# Proyecto-Multiescala

Este repositorio incluye el código del informe final del curso Procesamiento Multiescala de Imágenes, llamado Uso de algoritmos multiescala como regulizadores en inversión del dipolo en QSM

## Requisitos:
Wavelet Toolbox https://la.mathworks.com/products/wavelet.html
Parallel Computing Toolbox (opcional pero recomendado) https://la.mathworks.com/products/parallel-computing.html

## Códigos presentes:
run_demo.m Ejemplo de reconstrucción de fantoma con wavelets y con variación total (TV)
wWavelet.m Código para reconstruir QSM con regularización mediante wavelets
wTV.m Código para reconstruir QSM con Variación Total

## Material adicional:
spatial_res: tamaño de voxel
msk: máscara de cerebro
magn: imagen de magnitud
chi_cosmos: fantoma con COSMOS de 12 orientaciones

Ignacio Opazo Reyes
2025
