#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void binarizar(Mat src, Mat dst);
int etiquetado(Mat image, Mat &etiquetas, vector<int> &pixelesPorEtiqueta);
void filtroTamano(Mat image, vector<int> &pixelesPorEtiqueta, int numEtiqueta, float porcentaje, Mat etiquetas, Mat &imagenFiltradaTam, Mat &imagenFiltradaTamColoreada);
vector<Point> hallarCentros(Mat image, Mat etiquetas, Mat transDistancia, Mat &imagenCentros, vector<int> pixelesPorEtiqueta, int numEtiqueta);
vector<Point> obtener4Vecinos(Point punto, int cols, int rows);
bool pixelValido(Point punto, int cols, int rows);
void separarHuevos(Mat image, Mat &huevo, Point centro);

int main(int argc, char** argv){
    // Declare variables
    Mat src, crop, imageBin, image_dilate, etiquetas, imagenFiltradaTam, imagenFiltradaTamColoreada, transDistancia, imagenCentros;
    Mat huevos[5];
    Point anchor = Point(-1, -1);
    int numEtiqueta;
    vector<int> pixelesPorEtiqueta;
    vector<Point> centros;
    // Loads an image
    src = imread(argv[1], 1);
    if (src.empty()) {
        std::cerr << "Usage: " << argv[0] << " image_file" << std::endl;
        return -1;
    }
    // recortar imagen para eliminar las partes no deseadas al tomar la foto
    crop = src(Rect(80, 490, 760, 330));
    imageBin = Mat::zeros(crop.size(), CV_8UC1);
    // se binariza la imagen para segmentar los huevos
    binarizar(crop, imageBin);
    // se dilata la imagen para eliminar las caracteristicas no deseadas 
    image_dilate = Mat::zeros(imageBin.size(), CV_8UC1);
    dilate(imageBin, image_dilate, Mat(), anchor, 1, 1, 1);

    etiquetas = Mat::zeros(image_dilate.size(), CV_32S);
    imagenFiltradaTam = Mat::zeros(image_dilate.size(), CV_8UC1);
    imagenFiltradaTamColoreada = Mat::zeros(image_dilate.size(), CV_8UC1);

    // Etiquetar las regiones
    numEtiqueta = etiquetado(image_dilate, etiquetas, pixelesPorEtiqueta);

    // Filtrado por tamaño de la imagen
    filtroTamano(image_dilate, pixelesPorEtiqueta, numEtiqueta, 0.15, etiquetas, imagenFiltradaTam, imagenFiltradaTamColoreada);

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
    for (int i = 0; i < 5; i++)
        huevos[i]= Mat::zeros(image_dilate.size(), CV_8UC1);
    for (int i = 0, j=0; i < numEtiqueta && j<5; i++) {
        if (pixelesPorEtiqueta[i] != 0) {
            separarHuevos(image_dilate, huevos[j], centros[i]);
            j++;
        }
    }
    
    imwrite(basename + "_crop.png", crop);
    imwrite(basename + "_binarizada.png", imageBin);
    imwrite(basename + "_dilate.png", image_dilate);
    imwrite(basename + "_regionesFiltradasPorTamaño.png", imagenFiltradaTam);imwrite(basename + "_regionesFiltradasPorTamañoColoreada.png", imagenFiltradaTamColoreada);
    imwrite(basename + "_transDistancia.png", transDistancia);
    imwrite(basename + "_centros.png", imagenCentros);
    for(int i=0; i<5; ++i){
        imwrite(basename + "_huevo"+to_string(i)+".png", huevos[i]);
    }
    return 0;
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
    for (; it != end, itb != endb; ++it, ++itb) {
        if (!((*it)[2] < rUmbral && (*it)[1] > gUmbral)) {
            (*itb)[2] = 255;
            (*itb)[1] = 255;
            (*itb)[0] = 255;
        } else {
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
    } // rof
}

int etiquetado(Mat image, Mat &etiquetas, vector<int> &pixelesPorEtiqueta){
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
    cout << "Regiones obtenidas:" << endl;
    cout << "Region Pixeles" << endl;
    for (int i = 0; i < numEtiqueta; i++) {
        cout << setw(6) << i + 1 << setw(8) << pixelesPorEtiqueta[i] << endl;
    }
    return numEtiqueta;
}

void filtroTamano(Mat image, vector<int> &pixelesPorEtiqueta, int numEtiqueta, float porcentaje, Mat etiquetas, Mat &imagenFiltradaTam, Mat &imagenFiltradaTamColoreada){
    int regionMasGrande = 0;
    for (int i = 0; i < numEtiqueta; i++){
        if (pixelesPorEtiqueta[i] > regionMasGrande)
            regionMasGrande = pixelesPorEtiqueta[i];
    }
    cout << endl;
    cout << "Tamaño región más grande: " << regionMasGrande << endl;
    int filtro=porcentaje*regionMasGrande;
    cout << porcentaje * 100 << "% del tamaño región más grande: " << filtro << endl;
    int valorPixelBase = 255 / pixelesPorEtiqueta.size();
    cout << "Intensidad base: " << valorPixelBase << endl;
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            if (pixelesPorEtiqueta[etiquetas.at<int>(Point(i, j)) - 1] >= filtro) {
                imagenFiltradaTam.at<uchar>(Point(i, j)) = 255;
                imagenFiltradaTamColoreada.at<uchar>(Point(i, j)) = valorPixelBase * etiquetas.at<int>(Point(i, j));
            } else
                pixelesPorEtiqueta[etiquetas.at<int>(Point(i, j)) - 1] = 0;
        }
    }
    cout << endl;
    cout << "Después de filtrar por tamaño las regiones, las restastes son:" << endl;
    cout << "Region Pixeles" << endl;
    for (int i = 0; i < numEtiqueta; i++){
        if (pixelesPorEtiqueta[i] != 0)
            cout << setw(6) << i + 1 << setw(8) << pixelesPorEtiqueta[i] << endl;
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

    cout << endl;
    cout << "Centros de las regiones:" << endl;
    cout << "       X      Y" << endl;
    for (int i = 0; i < numEtiqueta; i++) {
        if (pixelesPorEtiqueta[i] != 0) {
            centros[i].x = centros[i].x / numPosiblesCentros[i];
            centros[i].y = centros[i].y / numPosiblesCentros[i];
            cout << setw(8) << centros[i].x << setw(8) << centros[i].y << endl;
            imagenCentros.at<uchar>(centros[i]) = 0;
        }
    }
    return centros;
}

void separarHuevos(Mat image, Mat &huevo, Point centro){
    Point left_up (centro.x-103, centro.y-80);
    cout<<centro<<endl;
    if(left_up.x<0)
        left_up.x=0;
    if(left_up.y<0)
        left_up.y=0;
    Point right_down (centro.x+90, centro.y+80);
    if(right_down.x>=image.rows)
        right_down.x=image.rows-1;
    if(right_down.y>=image.cols)
        right_down.y=image.cols-1;
    cout<<" "<<left_up<<" "<<right_down<<endl;
    huevo = image(Rect(left_up, right_down));
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