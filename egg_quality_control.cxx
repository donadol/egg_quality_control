#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkHistogramToTextureFeaturesFilter.h"
#include "itkScalarImageToCooccurrenceMatrixFilter.h"
#include "itkVectorContainer.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include <math.h>


//definitions of used types
typedef itk::Image<unsigned char, 2> InternalImageType;
typedef itk::Image<unsigned char, 2> VisualizingImageType;
typedef itk::Neighborhood<float, 2> NeighborhoodType;
typedef itk::Statistics::ScalarImageToCooccurrenceMatrixFilter<InternalImageType> Image2CoOccuranceType;
typedef Image2CoOccuranceType::HistogramType HistogramType;
typedef itk::Statistics::HistogramToTextureFeaturesFilter<HistogramType> Hist2FeaturesType;
typedef InternalImageType::OffsetType OffsetType;
typedef itk::AddImageFilter <InternalImageType> AddImageFilterType;
typedef itk::MultiplyImageFilter<InternalImageType> MultiplyImageFilterType;
typedef itk::RescaleIntensityImageFilter< InternalImageType, InternalImageType > RescaleFilterType;
typedef itk::RegionOfInterestImageFilter<InternalImageType,InternalImageType> roiType;
typedef itk::ImageFileWriter <InternalImageType> WriterType;


#define K_HIGH_R 241
#define K_LOW_R 163
#define K_HIGH_G 215
#define K_LOW_G 142
#define K_HIGH_B 192
#define K_LOW_B 115
#define K_HIGH_FORM 0.23
#define K_LOW_FORM 0.18
#define PI 3.14159265358979323846
#define K_INERTIA_O 6.252321976
#define K_INERTIA_1 4.948496536
#define K_INERTIA_2 6.238251917
#define K_INERTIA_3 3.825991037

using namespace cv;
using namespace std;

void binarizar(Mat src, Mat dst);
void segmentar(Mat src, Mat dst);
int etiquetado(Mat &image, Mat &etiquetas, vector<int> &pixelesPorEtiqueta);
int filtroTamano(Mat &image, vector<int> &pixelesPorEtiqueta, int numEtiqueta, float porcentaje, Mat etiquetas, Mat &imagenFiltradaTam, Mat &imagenFiltradaTamColoreada);
void filtroTamano(Mat &image, vector<int> &pixelesPorEtiqueta, int numEtiqueta, float porcentaje, Mat etiquetas, Mat &imagenFiltradaTam);
vector<Point> hallarCentros(Mat image, Mat etiquetas, Mat transDistancia, Mat &imagenCentros, vector<int> pixelesPorEtiqueta, int numEtiqueta);
vector<Point> obtener4Vecinos(Point punto, int cols, int rows);
bool pixelValido(Point punto, int cols, int rows);
Rect separarHuevos(Mat image, Mat &huevo, Point centro);
void treshGradiente(Mat src, Mat dst);
vector<float> promedio(Mat image, Mat imagebw);
void clasificar(Mat &image, Rect r, vector<float> prom, float numForma, bool grietas);
int totalPixeles(Mat image, Mat imagebw);
float numeroForma(Mat image, int pixeles);
void imagenOpenCVAITK (Mat &imageOCV, InternalImageType::Pointer imageITK);
bool tieneGrietas (Mat imageMat, int sizeX, int sizeY);
void colorearHuevo (Mat &bin_image, Mat &color_image, Mat &dilate_image);

int main(int argc, char** argv){
    // Declare variables
    Mat src, crop, imageBin, imageSeg, image_dilate, image_dilate_color,  etiquetas, imagenFiltradaTam, imagenFiltradaTamColoreada, transDistancia, imagenCentros, etiquetasHuevo, imagenClasificada;
    vector<Mat> huevos_color, huevos_bw, huevos_dist, huevos_gradiente;
    Point anchor = Point(-1, -1);
    int numEtiqueta, numEtiquetaFiltradas;
    vector<int> pixelesPorEtiqueta;
    vector<Point> centros;
    vector<Rect> rectHuevos;
    vector<float> aux;
    bool tieneGrietasB;
    // Revisar argumentos
    // imagen s -> se guardan todas las imagenes generadas en el proceso
    // imagen n -> se guarda solo la imagen final
    if (argc < 3){
        std::cerr << "Usage: " << argv[0] << " image_file s|n" << std::endl;
        return (-1);

    } // fi
    // Loads an image
    src = imread(argv[1], 1);
    if (src.empty()) {
        cerr << "Error: No image data" << endl;
        return -1;
    }
    // recortar imagen para eliminar las partes no deseadas al tomar la foto
    crop = src(Rect(70, 470, 780, 360));

    imageBin = Mat::zeros(crop.size(), CV_8UC1);
    imageSeg = Mat::zeros(crop.size(), CV_8UC3);
    // se binariza la imagen para separar los huevos del fondo
    binarizar(crop, imageBin);
    // se segmenta la imagen para separar los huevos del fondo
    segmentar(crop, imageSeg);
    // se dilata la imagen para eliminar las caracteristicas no deseadas
    image_dilate = Mat::zeros(imageBin.size(), CV_8UC1);
    dilate(imageBin, image_dilate, Mat(), anchor, 1, 1, 1);

    image_dilate_color = Mat::zeros(imageSeg.size(), CV_8UC3);
    dilate(imageSeg, image_dilate_color, Mat(), anchor, 1, 1, 1);

    //gradiente morfologico
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    etiquetas = Mat::zeros(image_dilate.size(), CV_32S);
    imagenFiltradaTam = Mat::zeros(image_dilate.size(), CV_8UC1);
    imagenFiltradaTamColoreada = Mat::zeros(image_dilate.size(), CV_8UC1);

    // Etiquetar las regiones
    numEtiqueta = etiquetado(image_dilate, etiquetas, pixelesPorEtiqueta);

    // Filtrado por tamaño de la imagen
    numEtiquetaFiltradas=filtroTamano(image_dilate, pixelesPorEtiqueta, numEtiqueta, 0.15, etiquetas, imagenFiltradaTam, imagenFiltradaTamColoreada);

    // Transformada distancia
    distanceTransform(imagenFiltradaTam, transDistancia, 2, 3);
    normalize(transDistancia, transDistancia, 0, 255, NORM_MINMAX);
    transDistancia.convertTo(transDistancia, CV_8U);

    // Hallar centro de cada huevo
    imagenCentros = imagenFiltradaTam.clone();
    centros=hallarCentros(image_dilate, etiquetas, transDistancia, imagenCentros, pixelesPorEtiqueta, numEtiqueta);
    std::stringstream ss(argv[1]);
    std::string basename;
    getline(ss, basename, '.');
    // Separar por huevos
    for (int i = 0; i < numEtiquetaFiltradas; i++){
        huevos_color.push_back(Mat::zeros(image_dilate.size(), CV_8UC3));
        huevos_bw.push_back(Mat::zeros(image_dilate.size(), CV_8UC1));
        huevos_gradiente.push_back(Mat::zeros(image_dilate.size(), CV_8UC1));
        huevos_dist.push_back(Mat::zeros(image_dilate.size(), CV_8UC1));
    }
    for (int i = 0, j=0; i < numEtiqueta && j<numEtiquetaFiltradas; i++) {
        if (pixelesPorEtiqueta[i] != 0) {
            rectHuevos.push_back(separarHuevos(imagenFiltradaTam, huevos_bw[j], centros[i]));
            huevos_color[j] = image_dilate_color(rectHuevos[j]).clone();

            // Remover ruido
            etiquetasHuevo=Mat::zeros(huevos_bw[j].size(), CV_32S);
            vector<int> pixelesPorEtiquetaHuevo;
            int n = etiquetado(huevos_bw[j], etiquetasHuevo, pixelesPorEtiquetaHuevo);
            Mat huevoPrueba = Mat::zeros(huevos_bw[j].size(), CV_8UC1);
            filtroTamano(huevos_bw[j], pixelesPorEtiquetaHuevo, n, 0.50, etiquetasHuevo, huevoPrueba );
            huevos_bw[j] = huevoPrueba;

            colorearHuevo (huevos_bw[j], huevos_color[j], huevos_color[j]);

            // Transformada distancia
            distanceTransform(huevos_bw[j], huevos_dist[j], 2, 3);
            normalize(huevos_dist[j], huevos_dist[j], 0, 255, NORM_MINMAX);
            huevos_dist[j].convertTo(huevos_dist[j], CV_8U);
            //huevos_dist[j] = transDistancia(rectHuevos[j]).clone();

            morphologyEx(huevos_color[j], huevos_gradiente[j], MORPH_GRADIENT, kernel);
            cvtColor (huevos_gradiente[j], huevos_gradiente[j], COLOR_RGB2GRAY);

            threshold( huevos_gradiente[j], huevos_gradiente[j], 25, 255, THRESH_BINARY);

            j++;
        }
    }
    imagenClasificada = crop.clone();
    for(int i=0; i<numEtiquetaFiltradas; i++){
        aux.clear();
        aux = promedio(huevos_color[i], huevos_bw[i]);
        cout<<"Huevo "<<i<<":"<<endl;
        cout<<"R: "<<aux[2]<<" G: "<<aux[1]<<" B: "<<aux[0]<<endl;
        float nf =numeroForma(huevos_dist[i], totalPixeles(huevos_dist[i], huevos_bw[i]));
        cout<<"Número de forma: "<<nf<<endl;
        tieneGrietasB = tieneGrietas(huevos_gradiente[i], huevos_gradiente[i].cols, huevos_gradiente[i].rows);
        clasificar(imagenClasificada, rectHuevos[i], aux, nf, tieneGrietasB);
        cout<<endl;
    }
    if(strcmp(argv[2], "s")==0){
        imwrite(basename + "_crop.png", crop);
        imwrite(basename + "_binarizada.png", imageBin);
        imwrite(basename + "_segmentada.png", imageSeg);
        imwrite(basename + "_dilate.png", image_dilate);
        imwrite(basename + "_dilate_color.png", image_dilate_color);
        imwrite(basename + "_regionesFiltradasPorTamaño.png", imagenFiltradaTam);
        imwrite(basename + "_regionesFiltradasPorTamañoColoreada.png", imagenFiltradaTamColoreada);
        imwrite(basename + "_transDistancia.png", transDistancia);
        imwrite(basename + "_centros.png", imagenCentros);
        for(int i=0; i<numEtiquetaFiltradas; ++i){
            imwrite(basename + "_huevo_gradiente_grises"+to_string(i)+".png", huevos_gradiente[i]);
            imwrite(basename + "_huevo_color"+to_string(i)+".png", huevos_color[i]);
            imwrite(basename + "_huevo_bw"+to_string(i)+".png", huevos_bw[i]);
            imwrite(basename + "_huevo_distancias"+to_string(i)+".png", huevos_dist[i]);
        }
    }
    imwrite(basename + "_clasificados.png", imagenClasificada);
    return 0;
}

void colorearHuevo (Mat &bin_image, Mat &color_image, Mat &dilate_image){
    for (int i = 0; i < bin_image.cols; i++ ){
        for (int j = 0; j< bin_image.rows; j++){
            if ((int)bin_image.at<uchar>(Point(i, j)) == 255 ){
                color_image.at<Vec3b>(Point(i, j)) = dilate_image.at<Vec3b>(Point(i, j));
            }
            else{
                color_image.at<Vec3b>(Point(i, j)) = {0, 0, 0};
            }
        }
    }
}

void binarizar(Mat src, Mat dst){
    int rHst[256] = { 0 }, gHst[256] = { 0 }, bHst[256] = { 0 };
    // Fill color channel images
    MatIterator_<Vec3b> it, end, itb, endb, itu, endu;
    it = src.begin<Vec3b>();
    end = src.end<Vec3b>();
    for (; it != end; ++it) {
        rHst[(*it)[2]]++;
        gHst[(*it)[1]]++;
        bHst[(*it)[0]]++;
    } // rof

    int rUmbral = 100;
    int gUmbral = 20;
    int bUmbral = 50;

    it = src.begin<Vec3b>();
    end = src.end<Vec3b>();
    itb = dst.begin<Vec3b>();
    endb = dst.end<Vec3b>();
    for (; it != end && itb != endb; ++it, ++itb) {
        (*itb)[2] = 255;
        (*itb)[1] = 255;
        (*itb)[0] = 255;


        if ((*it)[2] < 40 && (*it)[1] > 50 && (*it)[0] > 20) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if ((*it)[2] < 85 && (*it)[1] > 80 && (*it)[0] > 70) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

//VERDE OSCURO
        if ((*it)[2] < 30 && (*it)[1] > 20) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if ((*it)[2] < 50 && (*it)[1] > 60 && (*it)[1] > 60) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

//VERDE CLARO
        if ((*it)[2] < 250 && (*it)[1] > 250 && (*it)[1] > 250) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
        if (((*it)[2] < 25 && (*it)[1] < 25)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
        if (((*it)[2] < 200 && (*it)[1] > 130 && (*it)[0] > 160)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
        if (((*it)[2] < 130 && (*it)[1] > 125 && (*it)[0] <= 160)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }


        //PRUEBA
        if (((*it)[2] < 40 && (*it)[1] > 20 && (*it)[0] > 20)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if (((*it)[2] < 100 && (*it)[1] > 80 && (*it)[0] > 80)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if (((*it)[2] < 150 && (*it)[1] > 130 && (*it)[0] > 130)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
    } // rof
}

void segmentar(Mat src, Mat dst){
    int rHst[256] = { 0 }, gHst[256] = { 0 }, bHst[256] = { 0 };
    // Fill color channel images
    MatIterator_<Vec3b> it, end, itb, endb, itu, endu;
    it = src.begin<Vec3b>();
    end = src.end<Vec3b>();
    for (; it != end; ++it) {
        rHst[(*it)[2]]++;
        gHst[(*it)[1]]++;
        bHst[(*it)[0]]++;
    } // rof

    int rUmbral = 100;
    int gUmbral = 20;

    it = src.begin<Vec3b>();
    end = src.end<Vec3b>();
    itb = dst.begin<Vec3b>();
    endb = dst.end<Vec3b>();
    for (; it != end && itb != endb; ++it, ++itb) {
        (*itb)[2] = (*it)[2];
        (*itb)[1] = (*it)[1];
        (*itb)[0] = (*it)[0];

        if ((*it)[2] < 40 && (*it)[1] > 50 && (*it)[0] > 20) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if ((*it)[2] < 85 && (*it)[1] > 80 && (*it)[0] > 70) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

//VERDE OSCURO
        if ((*it)[2] < 30 && (*it)[1] > 20) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if ((*it)[2] < 50 && (*it)[1] > 60 && (*it)[1] > 60) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

//VERDE CLARO
        if ((*it)[2] < 250 && (*it)[1] > 250 && (*it)[1] > 250) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
        if (((*it)[2] < 25 && (*it)[1] < 25)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
        if (((*it)[2] < 200 && (*it)[1] > 130 && (*it)[0] > 160)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }
        if (((*it)[2] < 130 && (*it)[1] > 125 && (*it)[0] <= 160)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        //PRUEBA:
        if (((*it)[2] < 40 && (*it)[1] > 20 && (*it)[0] > 20)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if (((*it)[2] < 100 && (*it)[1] > 80 && (*it)[0] > 80)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

        if (((*it)[2] < 150 && (*it)[1] > 130 && (*it)[0] > 130)) {
            (*itb)[2] = 0;
            (*itb)[1] = 0;
            (*itb)[0] = 0;
        }

    } // rof
}

int etiquetado(Mat &image, Mat &etiquetas, vector<int> &pixelesPorEtiqueta){
    int numEtiqueta=0;
    queue<Point> cola;
    int valorPixel;
    vector<Point> vecinos;
    int valorPixelVecino;
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            valorPixel = (int)image.at<uchar>(Point(i, j));
            if (valorPixel == 255 && etiquetas.at<int>(Point(i, j)) == 0){
                numEtiqueta++;
                etiquetas.at<int>(Point(i, j)) = numEtiqueta;
                cola.push(Point(i, j));
                pixelesPorEtiqueta.push_back(0);
                while (!cola.empty()) {
                    Point pixelBlanco = cola.front();
                    cola.pop();
                    vecinos = obtener4Vecinos(pixelBlanco, image.cols, image.rows);
                    for (int k = 0; k < vecinos.size(); k++) {
                        valorPixelVecino = (int)image.at<uchar>(vecinos[k]);
                        if (valorPixelVecino == 255 && etiquetas.at<int>(vecinos[k]) == 0) {
                            etiquetas.at<int>(vecinos[k]) = numEtiqueta;
                            cola.push(vecinos[k]);
                            pixelesPorEtiqueta[numEtiqueta - 1]++;
                        }
                    }
                }
            }
        }
    }
    return numEtiqueta;
}

int filtroTamano(Mat &image, vector<int> &pixelesPorEtiqueta, int numEtiqueta, float porcentaje, Mat etiquetas, Mat &imagenFiltradaTam, Mat &imagenFiltradaTamColoreada){
    int regionMasGrande = 0;
    int n=0;
    for (int i = 0; i < numEtiqueta; i++){
        if (pixelesPorEtiqueta[i] > regionMasGrande)
            regionMasGrande = pixelesPorEtiqueta[i];
    }
    int filtro=porcentaje*regionMasGrande;
    int valorPixelBase = 255 / pixelesPorEtiqueta.size();
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            if (pixelesPorEtiqueta[etiquetas.at<int>(Point(i, j)) - 1] >= filtro) {
                imagenFiltradaTam.at<uchar>(Point(i, j)) = 255;
                imagenFiltradaTamColoreada.at<uchar>(Point(i, j)) = valorPixelBase * etiquetas.at<int>(Point(i, j));
            } else
                pixelesPorEtiqueta[etiquetas.at<int>(Point(i, j)) - 1] = 0;
        }
    }
    for (int i = 0; i < numEtiqueta; i++){
        if (pixelesPorEtiqueta[i] != 0)
            n++;
    }
    return n;
}

void filtroTamano(Mat &image, vector<int> &pixelesPorEtiqueta, int numEtiqueta, float porcentaje, Mat etiquetas, Mat &imagenFiltradaTam){
    int regionMasGrande = 0;
    for (int i = 0; i < numEtiqueta; i++){
        if (pixelesPorEtiqueta[i] > regionMasGrande)
            regionMasGrande = pixelesPorEtiqueta[i];
    }
    int filtro=porcentaje*regionMasGrande;
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            if (pixelesPorEtiqueta[etiquetas.at<int>(Point(i, j)) - 1] >= filtro) {
                imagenFiltradaTam.at<uchar>(Point(i, j)) = 255;
            } else
                pixelesPorEtiqueta[etiquetas.at<int>(Point(i, j)) - 1] = 0;
        }
    }
}

vector<Point> hallarCentros(Mat image, Mat etiquetas, Mat transDistancia, Mat &imagenCentros, vector<int> pixelesPorEtiqueta, int numEtiqueta){
    vector<int> mayoresValoresTransformada;
    vector<int> numPosiblesCentros;
    vector<Point> centros;
    int promedioX;
    int promedioY;

    for (int i = 0; i < numEtiqueta; i++)
        mayoresValoresTransformada.push_back(0);

    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            int etiquetaPixel = etiquetas.at<int>(Point(i, j));
            if (etiquetaPixel > 0) { //si es un pixel blanco etiquetado
                if ((int)transDistancia.at<uchar>(Point(i, j)) > mayoresValoresTransformada[etiquetaPixel - 1])
                    mayoresValoresTransformada[etiquetaPixel - 1] = (int)transDistancia.at<uchar>(Point(i, j));
            }
        }
    }

    for (int i = 0; i < numEtiqueta; i++) {
        numPosiblesCentros.push_back(0);
        centros.push_back(Point(0, 0));
    }

    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            int etiquetaPixel = etiquetas.at<int>(Point(i, j));
            if (etiquetaPixel > 0) {
                if ((int)transDistancia.at<uchar>(Point(i, j)) == mayoresValoresTransformada[etiquetaPixel - 1]) {
                    if (pixelesPorEtiqueta[etiquetaPixel - 1] != 0) {
                        promedioX = centros[etiquetaPixel - 1].x + i;
                        promedioY = centros[etiquetaPixel - 1].y + j;
                        numPosiblesCentros[etiquetaPixel - 1]++;
                        centros[etiquetaPixel - 1] = Point(promedioX, promedioY);
                    }
                }
            }
        }
    }
    for (int i = 0; i < numEtiqueta; i++) {
        if (pixelesPorEtiqueta[i] != 0) {
            centros[i].x = centros[i].x / numPosiblesCentros[i];
            centros[i].y = centros[i].y / numPosiblesCentros[i];
            imagenCentros.at<uchar>(centros[i]) = 0;
        }
    }
    return centros;
}

Rect separarHuevos(Mat image, Mat &huevo, Point centro){
    int right=0, left=0, up=0, down=0;
    for (int i = centro.x; i < image.cols-1; i++) {
        if((int)image.at<uchar>(Point(i+1, centro.y))==0){
            right=i;
            break;
        }
    }
    for (int i = centro.x; i > 0; i--) {
        if((int)image.at<uchar>(Point(i-1, centro.y))==0){
            left=i;
            break;
        }
    }
    for (int j = centro.y; j < image.rows-1; j++) {
        if((int)image.at<uchar>(Point(centro.x, j+1))==0){
            down=j;
            break;
        }
    }
    for (int j = centro.y; j > 0; j--) {
        if((int)image.at<uchar>(Point(centro.x, j-1))==0){
            up=j;
            break;
        }
    }
    Point left_up (left-20, up-20);
    if(left_up.x<0)
        left_up.x=0;
    if(left_up.y<0)
        left_up.y=0;
    Point right_down (right+20, down+20);
    if(right_down.x>=image.cols)
        right_down.x=image.cols-1;
    if(right_down.y>=image.rows)
        right_down.y=image.rows-1;
    Rect rect(left_up, right_down);
    huevo = image(rect);
    return rect;
}

vector<Point> obtener4Vecinos(Point punto, int cols, int rows){
    vector<Point> vecinos;
    int moves4[4][2] = { { -1, 0 }, { 0, -1 }, { 0, 1 }, { 1, 0 } };
    for (int i = 0; i < 4; i++) {
        Point vecino = Point(punto.x + moves4[i][0], punto.y + moves4[i][1]);
        if (pixelValido(vecino, cols, rows)) {
            vecinos.push_back(vecino);
        }
    }
    return vecinos;
}

bool pixelValido(Point punto, int cols, int rows){
    return (punto.x >= 0 && punto.x < cols) && (punto.y >= 0 && punto.y < rows);
}

void clasificar(Mat &image, Rect r, vector<float> prom, float numForma, bool grietas){
    bool pasa[3]={true, true, true};
    if(!((prom[2]<=K_HIGH_R && prom[2]>=K_LOW_R) && (prom[1]<=K_HIGH_G && prom[1]>=K_LOW_B) && (prom[0]<=K_HIGH_B && prom[0]>=K_LOW_B))){
        rectangle(image, r, Scalar(0, 0, 255),5,8,0);
        cout<<"Resultado: No pasa por color"<<endl;
        pasa[0]=false;
    }
    else 
        cout<<"Resultado: Pasa por color"<<endl;
    if(!(numForma<=K_HIGH_FORM && numForma >= K_LOW_FORM)){
        cout<<"Resultado: No pasa por forma"<<endl;
        rectangle(image, r, Scalar(0, 0, 255),5,8,0);
        pasa[1]=false;
    }
    else 
        cout<<"Resultado: Pasa por forma"<<endl;
    if(grietas){
        cout<<"Resultado: No pasa porque tiene grietas"<<endl;
        rectangle(image, r, Scalar(0, 0, 255),5,8,0);
        pasa[2]=false;
    }
    else
        cout<<"Resultado: Pasa porque tiene grietas"<<endl;
    
    if(pasa[0]==true && pasa[1]==true && pasa[2]==true){ 
        //cout<<"Resultado: Pasa todos los parametros"<<endl;
        rectangle(image, r, Scalar(0, 255, 0),5,8,0);
    }
}

vector<float> promedio(Mat image, Mat imagebw){
    vector<float> prom(3,0);
    int cont=0;
    MatIterator_<Vec3b> it, end;
    MatIterator_<uchar> itbw, endbw;
    it = image.begin<Vec3b>();
    end = image.end<Vec3b>();
    itbw = imagebw.begin<uchar>();
    endbw = imagebw.end<uchar>();
    for (; it != end && itbw!=end; ++it, ++itbw){
        if((*itbw)>0){
            prom[2] += (*it)[2];
            prom[1] += (*it)[1];
            prom[0] += (*it)[0];
            cont++;
        }
    }
    prom[2]=prom[2]/cont;
    prom[1]=prom[1]/cont;
    prom[0]=prom[0]/cont;
    return prom;
}

int totalPixeles(Mat image, Mat imagebw){
    int cont=0;
    MatIterator_<Vec3b> it, end;
    MatIterator_<uchar> itbw, endbw;
    it = image.begin<Vec3b>();
    end = image.end<Vec3b>();
    itbw = imagebw.begin<uchar>();
    endbw = imagebw.end<uchar>();
    for (; it != end && itbw!=end; ++it, ++itbw){
        if((*itbw)>0){
            cont++;
        }
    }
    return cont;
}

float numeroForma(Mat image, int pixeles){
    int cont=0;
    MatIterator_<uchar> it, end;
    it = image.begin<uchar>();
    end = image.end<uchar>();
    for (; it != end; ++it){
        cont+=(*it);
    }
    return (float)pow(pixeles,3)/(float)(9*PI*pow(cont-sqrt(0.5)/2*cont,2));
}

void imagenOpenCVAITK (Mat &imageOCV, InternalImageType::Pointer imageITK){
    InternalImageType::IndexType start;
    start[0] = 0;  // first index on X
    start[1] = 0;  // first index on Y

    InternalImageType::SizeType  size;
    size[0] = imageOCV.cols;  // size along X
    size[1] = imageOCV.rows;  // size along Y

    InternalImageType::RegionType region;
    region.SetSize( size );
    region.SetIndex( start );

    imageITK->SetRegions( region );
    imageITK->Allocate();

    Mat_<uchar>::iterator itMat= imageOCV.begin<uchar>();
// obtain end position
    Mat_<uchar>::iterator itend= imageOCV.end<uchar>();

    itk::ImageRegionIterator<InternalImageType> it (imageITK,region);
    for (; !it.IsAtEnd(); ++it, itMat++){
        it.Set( (int)(*itMat) );
    }

    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName("ImagenGRISITK.png");
    writer->SetInput(imageITK);
    writer->Update();

}



bool tieneGrietas (Mat imageMat, int sizeX, int sizeY){
    InternalImageType::Pointer inputImage = InternalImageType::New();
    imagenOpenCVAITK(imageMat, inputImage);

    NeighborhoodType neighborhood;
    neighborhood.SetRadius( 1 );
    unsigned int centerIndex = neighborhood.GetCenterNeighborhoodIndex( );

    OffsetType offset;

    for ( unsigned int d = 0; d < centerIndex; d++ )
    {
      offset = neighborhood.GetOffset( d );

      std::cout << "Orientation " << d << ": ";

      Image2CoOccuranceType::Pointer glcmGenerator = Image2CoOccuranceType::New( );
      glcmGenerator->SetOffset( offset );
      glcmGenerator->SetNumberOfBinsPerAxis( 16 ); //reasonable number of bins
      glcmGenerator->SetPixelValueMinMax( 0, 255 ); //for input UCHAR pixel type

      Hist2FeaturesType::Pointer featureCalc = Hist2FeaturesType::New( );

      //region of interest
      roiType::Pointer roi = roiType::New( );
      roi->SetInput( inputImage );

      InternalImageType::IndexType start;
      start[0] = 0;
      start[1] = 0;

      InternalImageType::IndexType end;
      end[0] = sizeX - 1;
      end[1] = sizeY - 1;

      InternalImageType::RegionType region;
      region.SetIndex( start );
      region.SetUpperIndex( end );

      roi->SetRegionOfInterest( region );
      roi->Update( );

      glcmGenerator->SetInput( roi->GetOutput( ) );
      glcmGenerator->Update( );

      featureCalc->SetInput( glcmGenerator->GetOutput( ) );
      featureCalc->Update( );


      double inertia = featureCalc->GetInertia( );
      cout << "inertia: " << inertia<<endl;
      if (d == 0)
          if (inertia > K_INERTIA_O)
            return true;

      if (d == 1)
        if (inertia > K_INERTIA_1)
            return true;
      if (d == 2)
        if (inertia > K_INERTIA_2)
            return true;
      if (d == 3)
         if (inertia > K_INERTIA_3)
            return true;
  }
  return false;
}
