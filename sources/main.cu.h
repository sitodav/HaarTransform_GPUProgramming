#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <cuda.h>
#include <sstream>


using namespace std;
using namespace cv;

int mostraImmagineFlag;

/*classe che incapsula utilizzo degli eventi per il timing */

class MioTimer{
	cudaEvent_t startEvent,stopEvent;
	bool started;
	
	public:
		MioTimer(){
			started = false;
			cudaEventCreate(&startEvent);
			cudaEventCreate(&stopEvent);
		}
		
		void start(){
			if(started)
				return;
			cudaEventRecord(startEvent); //manda nello stream di lavoro (ordine FIFO ) del device
			started = true;
		}
		
		float stopECalcolaTempo(){
			if(!started){
				cout<<"timer non startato"<<endl;
				return 0;
			}
			//mandiamo nello stream di lavoro secondo evento
			cudaEventRecord(stopEvent); 
			//aspettiamo che nello stream di lavoro del device sia registrato l'evento
			cudaEventSynchronize(stopEvent);
			
			float elapsed = 0 ;
			cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
			
			started = false;
			return elapsed;
		}
		
		~MioTimer(){
			cudaEventDestroy(startEvent);
			cudaEventDestroy(stopEvent);
		}
};



/*in generale si lavora con mat di float (CV_32FC1) che sta a significare un singolo canale (perchè immagine aperta come grayscale)
con intensità memorizzate come float a 4 bytes. Quindi ci saranno valori tra 0 e 255 ma FLOAT !
Non usiamo il formato CV_8UC1 poichè perdiamo troppa informazione nell'arrotondamento !!
Tuttavia non si possono visualizzare Mat di tipo CV_32FC1, a meno che non si voglia prima normalizzare i valori tra 0 e 1 (cosa che non facciamo). 
Quindi per poter visualizzare i passi intermedi, si copiano i valori in matrici CV_8UC1
che pero' vengono solo usate per la visualizzazione, mentre i passaggi di trasformazioni e ricostruzioni sono effettuati sempre con le Mat di tipo CV_32FC1 */


__host__ void mostraImmagine(const Mat & img, char *text){
	if(mostraImmagineFlag == 0)
		return;
	//questa routine quindi prende una mat di valori float (che non è visualizzabile poichè sono float ma non normalizzati, mentre opencv o vuole float normalizzati tra 0 e 1, o vuole interi tra 0 e 255)
	//e converte in immagine con pixel di tipo CV_8UC1 per la visualizzazione
	Mat toShow(img.rows,img.cols,CV_8UC1);
	img.convertTo(toShow,CV_8UC1);
	imshow(text,toShow);
	waitKey(0);
	
}

__host__ void salvaImmagine(const Mat & img, const char * nome){
	imwrite(nome, img);
}


/*--------------------------check errori*/
void checkErroreCuda(char *msg){
	cudaError_t err;
	if((err=cudaGetLastError())!=cudaSuccess){
		cout<<msg<<" : "<<cudaGetErrorString(err)<<endl;
	}
	
}



/*routine per il calcolo trasformata in sequenziale sull'host ----------------------------------------------*/
__host__ void haarTransformHost(const Mat &img,Mat outImg,float coeff){ /*routine per la trasformata di Haar usando la mappatura tra pixel e quadrante di riferimento */
	Mat out(img.rows,img.cols,CV_32FC1);
	
	for(int i=0;i<out.rows;i++)
		for(int j=0;j<out.cols;j++){
			int bigI = i % (img.rows/2);
			int bigJ = j % (img.cols/2);
			
			float a = (float)img.at<float>(bigI * 2, bigJ * 2);
			float b = (float)img.at<float>(bigI * 2, 1 + bigJ * 2);
			float c = (float)img.at<float>(1+bigI * 2, bigJ * 2);
			float d = (float)img.at<float>(1+bigI * 2, 1+bigJ * 2);
			
			 
			
			if( (i / (img.rows/2) ) < 1 ){
				
				if( j / (img.cols/2) < 1 )
					out.at<float>(i,j)= coeff* (d*coeff + b*coeff) + (coeff) * (c*coeff + a*coeff); 
				
				else
					out.at<float>(i,j)= (coeff) * (d*coeff + b*coeff) - (coeff) * (c*coeff + a*coeff);
				
			}
			
			else{
				
				
				if( j / (img.cols/2) < 1 )
						out.at<float>(i,j)= (coeff) * (d*coeff - b*coeff) + (coeff) * (c*coeff - a*coeff);
				else
						out.at<float>(i,j)= (coeff) * (d*coeff - b*coeff) - (coeff) * (c*coeff - a*coeff);
				
			}
		} 
		
		out.convertTo(outImg,CV_32FC1);
		
}

__host__ void haarTransformHostV2(const Mat &img,Mat outImg,float coeff){ /*routine per la trasformata di Haar come convoluzione prima sulle colonne e poi sulle righe */
		Mat out(img.rows,img.cols,CV_32FC1);
		Mat out2(img.rows,img.cols,CV_32FC1);
		
		int m = img.rows;
		int n = img.cols;
		
		for(int j=0;j<img.cols;j++){
			int k=0;
			for(int i=0;i<img.rows;i++){
				if(i<(m/2))
					out.at<float>(i,j) = coeff * (img.at<float>(k % m,j) + img.at<float>((k+1)%m,j));
				else 
					out.at<float>(i,j) = coeff * (-img.at<float>(k % m,j) + img.at<float>((k+1)%m,j));
				k+=2;
			}
		}
		
		for(int i=0;i<m;i++){
			int k=0;
			for(int j=0;j<n;j++){
					if(j<(n/2))
						out2.at<float>(i,j) = coeff * (out.at<float>(i,k%n) + out.at<float>(i,(k+1)%n ) );
					else 
						out2.at<float>(i,j) = coeff * (-out.at<float>(i,k%n) + out.at<float>(i,(k+1)%n ) );
					k+=2;
			}
		}
		
		out2.convertTo(outImg,CV_32FC1);
		

}

__host__ void callRecursiveHaarHost(const Mat img,Mat finalRes,int & lvl,float coeff){ /* routine per chiamare ricorsivamente più livelli della trasformata di Haar */
//si presuppone che l'immagine che viene passata come parametro sia divisibile almeno 1 volta per 2
		MioTimer timer;							 
		int actLvl=0;
		Mat *res,*actual= new Mat(img);
		
		 
		/*start conteggio tempo */
		timer.start();
		while(++actLvl<=lvl){
			
			//cout<<"chiamata per trasformata di Haar su CPU livello "<<actLvl<<endl; 
			
			res = new Mat(actual->rows,actual->cols,CV_32FC1);
			haarTransformHostV2(*actual,*res,coeff);
			
			//copio res in finalRes
			for(int i=0;i<res->rows;i++)
				for(int j=0;j<res->cols;j++)
					finalRes.at<float>(i,j) = res->at<float>(i,j); 
			
			
			
			//mostraImmagine(finalRes,"Trasformata di Haar");
			
			
			if ( (res->rows/2) % 2 != 0 || (res->cols/2) % 2 != 0 ){ //non posso continuare a dividere
				lvl = actLvl;
				//cout<<"impossibile continuare col nesting! "<<res->rows<< " " <<res->cols<<endl;
				break;
			}
			
			//creo una nuova matrice grande quanto il quadrante DC di res che ne contiene i pixel, e questa diventa il nuovo actual
			actual = new Mat(res->rows/2,res->cols/2,CV_32FC1);
			for(int i=0;i<actual->rows;i++)
				for(int j=0;j<actual->cols;j++)
					actual->at<float>(i,j) = res->at<float>(i,j);
			
		}
		
		/*stop calcolo tempo */
		cout<<"tempo trascorso per calcolare trasformata di livello "<<lvl<<" usando CPU : "<<timer.stopECalcolaTempo()<<" ms"<<endl;
		
		

} 

/*-----------------------------------------------------------------------------------------------------------*/



/*routine per la ricostruzione sequenziale immagine originale da trasformata di haar (conosciuto il livello di nesting)---------------------------------------------------------*/
__host__ void reverseTransformHostV2(const Mat &img,Mat outImg,float coeff){ /*routine per ricostruzione immagine originale data una trasformata di haar, per un solo livello  */
		Mat out(img.rows,img.cols,CV_32FC1);
		Mat out2(img.rows,img.cols,CV_32FC1);
		
		int m = img.rows;
		int n = img.cols;
		
		for(int i=0;i<m;i++){
			
			for(int j=0;j<n/2;j++){
				out.at<float>(i,j*2) = coeff * img.at<float>(i,j) - coeff * img.at<float>(i,j+(n/2)); 
				out.at<float>(i,1+j*2) = coeff * img.at<float>(i,j) + coeff * img.at<float>(i,j+(n/2));
			}
		}
		
		for(int j=0;j<n;j++){
			for(int i=0;i<m/2;i++){
				out2.at<float>(i*2,j) = coeff * out.at<float>(i,j) - coeff * out.at<float>(i+(m/2),j);
				out2.at<float>(1+i*2,j) = coeff * out.at<float>(i,j) + coeff * out.at<float>(i+(m/2),j);
			}
		}
		
		out2.convertTo(outImg,CV_32FC1);
		
}

/*routine per la ricostruzione immagine originale per più livelli */
__host__ void callRecursiveRicostr(const Mat & img,Mat & finalRes,int & lvl,float coeff){ //si ipotizza che prende in input immagine ottenuta da lvl successive trasformazioni di haar
	 
		
		int startingM = img.rows,startingN = img.cols;
		Mat *toReverse,*reversed;
		int actLvl = 0;
		
		//per prima cosa copio in finalRes il contenuto della matrice da ricostruire
		for(int i=0;i<img.rows;i++)
			for(int j=0;j<img.cols;j++)
				finalRes.at<float>(i,j)=img.at<float>(i,j);
		
		
		for(int i=1;i<=lvl;i++){ 
			startingM = startingM / 2; 
			startingN = startingN /2;
		}
 
		 
 
		while(++actLvl<=lvl){
			startingM = startingM * 2;
			startingN = startingN * 2;
			//cout<<"reverse per "<<startingM<<" "<<startingN<<endl;
			toReverse = new Mat(startingM,startingN,CV_32FC1);
			
			
			for(int i=0;i<toReverse->rows;i++)
				for(int j=0;j<toReverse->cols;j++)
					toReverse->at<float>(i,j) = finalRes.at<float>(i,j);
			
			reversed = new Mat(toReverse->rows,toReverse->cols,CV_32FC1);
			reverseTransformHostV2(*toReverse,*reversed,coeff);
			for(int i=0;i<reversed->rows;i++)
				for(int j=0;j<reversed->cols;j++)
					finalRes.at<float>(i,j) = reversed->at<float>(i,j);
			
			
		
			
		}
}
/*-----------------------------------------------------------------------------------------------------------*/




/*kernel per ridisporre elementi dell'array contenente l'immagine, in parallelo----------------------------*/
__global__ void ridisponiArray(float *imgDataOriginaleD,float *imgDataRidispostoD,int m,int n){
	int globId = blockIdx.x * blockDim.x + threadIdx.x;
	if(globId >= m*n)
		return;
	
	int idQuad = globId / 4;
	int bigI = idQuad / (n/2);
	int bigJ = idQuad % (n/2);
	int indInSottogruppo = globId % 4;
	int iSottogruppo = indInSottogruppo / 2;
	int jSottogruppo = indInSottogruppo % 2;
	
	
	float elToCopy = imgDataOriginaleD[(2*bigI + iSottogruppo)*n + 2*bigJ + jSottogruppo];
	imgDataRidispostoD[globId] = elToCopy;
}

/*kernel per la trasformata di haar in parallelo -----------------------------------------------------------*/
__global__ void trasformataHaarGPU(float * imgDataD,float * trasformataDataD,int m,int n,float coeff){ //m numero righe immagine, n colonne
	extern __shared__ float shared_buffer[]; //questo viene allocato proprio per un numero di float pari a blockDim.x
	
	int globId = blockIdx.x * blockDim.x + threadIdx.x; //id del thread relativamente a tutta la griglia
	
	if(globId >= m*n)
		return;
	
		
	//devo capire quel thread a quale quadrupla di pixel appartiene
	int idQuad = globId / 4;
	//ciascun thread copia in un elemento della shared memory un singolo pixel dall'immagine
	shared_buffer[threadIdx.x] = imgDataD[globId];
	__syncthreads();
	//ciascun thread, relativamente al sottogruppo di 4 thread (nel blocco) ha un ruolo
	//a seconda del quadrante (dell'immagine finale) relativamente al quale effettua il calcolo
	//quest'informazione è ottenibile come modulo tra globId e 4
	int idSottogruppo = globId % 4;
	int val; 
	switch(idSottogruppo){
		case 0:
			val = coeff * (coeff * shared_buffer[threadIdx.x] + coeff * shared_buffer[threadIdx.x+2] ) + coeff * (coeff * shared_buffer[threadIdx.x+1] + coeff * shared_buffer[threadIdx.x+3]);
			break;
		case 1:
			val = -coeff * (coeff * shared_buffer[threadIdx.x+1] + coeff * shared_buffer[threadIdx.x-1] ) + coeff * (coeff * shared_buffer[threadIdx.x+2] + coeff * shared_buffer[threadIdx.x]);
			break;
		case 2:
			val = coeff * (coeff * shared_buffer[threadIdx.x] - coeff * shared_buffer[threadIdx.x-2] ) + coeff * (coeff * shared_buffer[threadIdx.x+1] - coeff * shared_buffer[threadIdx.x-1]);
			break;
		case 3:
			val = -coeff * (coeff * shared_buffer[threadIdx.x-1] - coeff * shared_buffer[threadIdx.x-3] ) + coeff * (coeff * shared_buffer[threadIdx.x] - coeff * shared_buffer[threadIdx.x-2]);
			break;
	}
	//per sapere in quale pixel dell'array output (rappresentante l'immagine trasformata) va messo val,
	//devo guardare all'indice riga e indice colonna, della quadrupla di pixel, che calcolo in questo modo
	int bigI = idQuad / (n/2);
	int bigJ= idQuad % (n/2);
	
	//e calcolo indice riga e colonna, del thread, all'interno del blocco di 4 thread, come se fosse un quadrato 2x2
	int iSottogruppo = idSottogruppo/2;
	int jSottogruppo = idSottogruppo % 2;
	//quindi il thread deve mettere, nel vettore monodimensionale rappresentante immagine di output, il valore in
	
	trasformataDataD[bigI * n + iSottogruppo * (m/2)*n + bigJ + jSottogruppo * (n/2)]=val;
	
}


/*NB: le due routine callRecursiveHaarDeviceCopieIntermedie e callHaarDeviceOneStep prevedono che per ogni livello di nesting, i sottoquadranti vengano trasferiti da memoria host a memoria device. In modo tale da poter visualizzare
sull'host i passi intermedi. Questo approccio tuttavia è estremamente dispendioso !! */

/*routine che prende in input immagine da trasformare, setta griglia e chiama kernel per la trasformata in parallelo su device*/
/*e ritorna puntatore a Mat contenente la trasformazione */
__host__ Mat * callHaarDeviceOneStep(const Mat & img, const dim3 & sizeBlocco, const dim3 & sizeGriglia,float coeff){
	
	//devo innanzitutto ricopiare i pixel immagine, in un vettore monodimensionale, pero' stendendo in maniera sequenziale le quadruple di pixel (quindi 0,0-0,1-1,0-1,1-0,2-0,3-1,2-1,3 ...)
	//per fare questo uso un kernel apposito
	float * imgDataOriginaleH = (float *)img.data; //questo è l'array monodimensionale classico dell'immagine
	float *imgDataOriginaleD ; //questo lo alloco sul device, e conterrà la copia dell'array monodimensionale classico da ridisporre
	float *imgDataRidispostoD; //questo conterrà, sul device, l'array ridisposto
	
	//alloco su device lo spazio dove copiare l'array originale 
	cudaMalloc(&imgDataOriginaleD,img.rows*img.cols*sizeof(float));
	checkErroreCuda("imgDataOriginaleD cudaMalloc");
	//copio il contenuto dall'host al device
	cudaMemcpy(imgDataOriginaleD,imgDataOriginaleH,img.rows*img.cols*sizeof(float),cudaMemcpyHostToDevice);
	//alloco l'array sul device che conterrà l'array ridisposto
	cudaMalloc(&imgDataRidispostoD,img.rows*img.cols*sizeof(float));
	checkErroreCuda("imgDataRidispostoD cudaMalloc");
	
	//lancio kernel
	ridisponiArray<<<sizeGriglia,sizeBlocco>>>(imgDataOriginaleD,imgDataRidispostoD,img.rows,img.cols);
	checkErroreCuda("chiamata kernel ridisposizione elementi immagine");
	cudaThreadSynchronize();
	//ora in imgDataRidispostoD c'e' l'array, ridisposto in maniera tale da garantire, al kernel che effettuerà la trasformata di Haar, la lettura coalescente
	//quindi tale kernel lavorerà direttamente su questo spazio di memoria device, senza che sia necessario ricopiarlo sull'host
	
	/* 
	
	//trasformazione direttamente in sequenziale	
	for(int bigI=0;bigI< (img.rows/2); bigI++)
		for(int bigJ = 0; bigJ<img.cols/2; bigJ++ ){
			int offset= (bigI * (img.cols / 2) + bigJ) * 4;
			imgDataRidispostoH[offset++] = img.at<float>( 2*bigI, 2*bigJ );
			imgDataRidispostoH[offset++] = img.at<float>(2*bigI, 1+2*bigJ);
			imgDataRidispostoH[offset++] = img.at<float>(1+ 2*bigI, 2*bigJ);
			imgDataRidispostoH[offset] = img.at<float>(1+ 2*bigI, 1+ 2*bigJ);
		} */
	 
	 
	float *trasformataDataD; //questo conterrà il risultato (sul device)
	//cudaMalloc(&imgDataD,img.rows*img.cols*sizeof(float)); //allochiamo e copiamo su device array da trasformare
	checkErroreCuda("imgDataD malloc");
	
	cudaMalloc(&trasformataDataD,img.rows * img.cols * sizeof(float)); checkErroreCuda("cudaMalloc trasformataDataD");
	cudaMemset(trasformataDataD,0,img.rows*img.cols*sizeof(float)); checkErroreCuda("trasformataDataD");
	//chiamo kernel trasformata di haar
	trasformataHaarGPU<<<sizeGriglia,sizeBlocco,sizeBlocco.x * sizeof(float)>>>(imgDataRidispostoD,trasformataDataD,img.rows,img.cols,coeff); //per ciascun thread, si alloca dinamicamente (su heap memoria gpu) un buffer shared memory di un numero di elementi (float) pari al numero di thread per blocco
	checkErroreCuda("chiamata kernel trasformata di haar");
	cudaThreadSynchronize();
	//copio risultato sull'host
	float *trasformataDataH = (float *)malloc(img.rows * img.cols * sizeof(float)); //alloco su host array che conterrà risultato della trasformata di haar
	
	cudaMemcpy(trasformataDataH,trasformataDataD,img.rows*img.cols*sizeof(float),cudaMemcpyDeviceToHost);
	checkErroreCuda("cudaMemcpy trasformataDataH");
	
	//creo l'immagine dall'array monodimensionale
	
	/*  Mat *imgTrasformata = new Mat(img.rows,img.cols,CV_32FC1);
	for(int i=0;i<img.rows;i++)
		for(int j=0;j<img.cols;j++)
			imgTrasformata->at<float>(i,j) = trasformataDataH[i*img.cols + j];  */
		
	Mat *imgTrasformata = new Mat(img.rows,img.cols,CV_32FC1,trasformataDataH); 
	
	//mostraImmagine(*imgTrasformata,"Trasformata Device");
	
	
	//cudaFree(imgDataD);
	cudaFree(trasformataDataD);
	cudaFree(imgDataOriginaleD);
	cudaFree(imgDataRidispostoD);
	//free(imgDataRidispostoH);
	//free(trasformataDataH); //non posso liberare perchè in imgTrasformata la mat fa shallow copy
	
	return imgTrasformata;
	
}
/*-----------------------------------------------------------------------------------------------------------*/

//routine per gestire livelli di nesting ricorsivo della trasformata di haar, prende in input immagine da trasformare---------------------------------------------------------
__host__ Mat * callRecursiveHaarDeviceCopieIntermedie (const Mat & img,int & lvl, const dim3 & sizeBlocco, const dim3 & sizeGriglia, float coeff){
	MioTimer timer;
	int actualLvl = 1;
	int actualM=img.rows, actualN= img.cols;
	Mat *finalResult = new Mat(img.clone()); //img.clone() ritorna MATRICE COPIA di img, mentre il copy constructor di una mat (cosi come il = operator) effettua una shallow copy (la matrice viene condivisa) 
											//quindi sto in pratica sto allocando dinamicamente una mat ottenuta con shallow copy di una nuova matrice copia di img
	Mat toTransform;
	Mat *transformed;
	Mat toShow(img.rows,img.cols,CV_8UC1);
	
	
	//NB quindi: in open cv sia il copy constructor tra mat che l'operatore = tra mat, creano un nuovo header, cioè ad esempio: 
	// A= B    allora A e B puntano alla stessa matrice, mentre
	// A= B (range(), range())   // allora A punta alla sottomatrice in B, se si fa 
	// C.copyTo(A) viene copiato nella sottomatrice di B (che è puntata da A) il contenuto di quanto è in C
	
	/*start calcolo tempi */
	timer.start();
	
	while(actualLvl <= lvl){
			//cout<<"chiamata per trasformata di Haar su GPU livello "<<actualLvl<<endl;
			if(actualM % 2 != 0 || actualN % 2 != 0){
				//cout<<"impossibile continuare col nesting!"<<actualM<< " " <<actualN<<endl;
				lvl = actualLvl-1; //in questo modo, non si tenta di fare una ricostruzione su n livelli se in realtà la trasformata è stata fatta per un numero minore di livelli (-1 perchè in gpu il break è fatto ad un passo successivo rispetto versione host)
				break;
			}
			toTransform = ((*finalResult)(Range(0,actualM), Range(0,actualN))).clone() ;    
			transformed = (Mat *) callHaarDeviceOneStep(toTransform,sizeBlocco,sizeGriglia,coeff); 
			for(int i=0;i<transformed->rows;i++)
				for(int j=0;j<transformed->cols;j++)
					finalResult->at<float>(i,j) = transformed->at<float>(i,j);
			
			
			
			actualM = actualM / 2;
			actualN = actualN/2;
			
			//mostraImmagine(*finalResult,"Trasformata Haar su Device");
			
			
			actualLvl++;
			
	}
	
	cout<<"tempo: livello trasformata: "<<lvl<<" usando GPU , numero threads per blocco: "<<sizeBlocco.x<<" con cudaMemcpy intermedie : "<<timer.stopECalcolaTempo()<<" ms"<<endl;

	
	return finalResult;
}



//qui ci sono le routine per calcolare la trasformata di haar senza le copie (per i nesting intermedi) da memoria device a host e viceversa
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
//questo kernel è una leggera modifica del kernel che ridispone gli elementi quando si usano le routine per i passi intermedi, solo che questa
//tiene conto del fatto che l'array che rappresenta l'immagine in memoria ha uno stride delle colonne diverso da quello su cui si sta attualmente lavorando
//Quindi questa routine prende un'array, le dimensioni del sottoquadrante su cui lavorare. L'array mantiene i pixel secondo la disposizione classica di allocazione per righe di un'immagine 
//la routine considera solo il sottoquadrante e ridispone gli elementi (a 4 a 4) per la lettura coalescente, 
__global__ void ridisponiPorzione(float *imgDataDaRidisporre, float *imgDataRidispostoD, int m,int n,int realM,int realN){
	int globId = blockIdx.x * blockDim.x + threadIdx.x;
	 
	
	int idQuad = globId / 4;
	int bigI = idQuad / (realN/2);
	int bigJ = idQuad % (realN/2);
	if(bigI >= (m / 2) || bigJ >= (n / 2) ) //in tal caso il thread in questione è assegnato ad un blocco appartenente ad una sottoimmagine su cui non dobbiamo lavorare al livello di nesting attuale
		return;
	int indInSottogruppo = globId % 4;
	int iSottogruppo = indInSottogruppo / 2;
	int jSottogruppo = indInSottogruppo % 2;
	
	
	float elToCopy = imgDataDaRidisporre[(2*bigI + iSottogruppo)*realN + 2*bigJ + jSottogruppo]; //questo è l'elemento da ridisporre
	imgDataRidispostoD[globId] = elToCopy; //lo mettiamo nell'array degli elementi ridisposti, all'indice uguale all'id nella griglia del thread in modo tale che possa essere recuperato all'interno del kernel per la trasformata
	//si noti che visto che si sta lavorando su un sottoquadrante dell'array allocato per l'immagine originale, se si considera l'intera griglia di thread, 
	//le quadruple (gruppi logici) di thread che lavoreranno saranno quelle relative al solo sottoquadrante di interesse.
	/*ad esempio per un'immagine 
	
	A B X X
	C D X X
	X X X X 
	dove il sottoblocco di lavoro (lettere diverse da X) dove A B C D sono blocchi di 4 pixel ciascuno , per ciascuno dei quali lavorano 4 threads
	avremo la disposizione A B X X C D X X X X X X (i blocchi di 4 pixel sono sempre disposti per righe, la ridisposizione è stata effettuata a livello dei pixel interni)
	e quindi lavoreranno i thread appartenenti ai gruppi logici di 4 thread in A B C D. Tuttavia la lettura coalescente è comunque rispettata. */
}

//questo kernel prende l'array degli elementi ridisposti (relativi al sottoquadrante del livello di nesting attuale, identificato dalle dimensioni m x n ) ne fa la trasformata, ed infine mette il solo sotto-sottoquadrante per il livello successivo di nesting
// ( fatto di pixel che andranno ridisposti al giro successivo) nell'array nextQuad 
__global__ void trasformataHaarGPUv2(float *imgDataRidispostoD,float * trasformataDataD,float *nextQuad,int m,int n,int realM,int realN,float coeff){
	extern __shared__ float shared_buffer[]; 
	
	int globId = blockIdx.x * blockDim.x + threadIdx.x; //id del thread relativamente a tutta la griglia
	
	 
	
 	int idQuad = globId / 4;
	int bigI = idQuad / (realN/2);
	int bigJ= idQuad % (realN/2);
	//e calcolo indice riga e colonna, del thread, all'interno del blocco di 4 thread, come se fosse un quadrato 2x2
	int idSottogruppo = globId % 4;
	int iSottogruppo = idSottogruppo/2;
	int jSottogruppo = idSottogruppo % 2;
	
	if(bigI >= (m / 2) || bigJ >= (n / 2) ) //anche qui, se il thread è assegnato ad un quadrante fuori dal sottoblocco di interesse (al livello di nesting attuale) non deve continuare
		return;
 
	shared_buffer[threadIdx.x] = imgDataRidispostoD[globId]; //recuperiamo il pixel che era stato ridisposto secondo la disposizione dei thread in quadranti consecutivi di quadruple consecutive (per lettura coalescente)
	__syncthreads();
	 
	
	int val; 
	switch(idSottogruppo){
		case 0:
			val = coeff * (coeff * shared_buffer[threadIdx.x] + coeff * shared_buffer[threadIdx.x+2] ) + coeff * (coeff * shared_buffer[threadIdx.x+1] + coeff * shared_buffer[threadIdx.x+3]);
			break;
		case 1:
			val = -coeff * (coeff * shared_buffer[threadIdx.x+1] + coeff * shared_buffer[threadIdx.x-1] ) + coeff * (coeff * shared_buffer[threadIdx.x+2] + coeff * shared_buffer[threadIdx.x]);
			break;
		case 2:
			val = coeff * (coeff * shared_buffer[threadIdx.x] - coeff * shared_buffer[threadIdx.x-2] ) + coeff * (coeff * shared_buffer[threadIdx.x+1] - coeff * shared_buffer[threadIdx.x-1]);
			break;
		case 3:
			val = -coeff * (coeff * shared_buffer[threadIdx.x-1] - coeff * shared_buffer[threadIdx.x-3] ) + coeff * (coeff * shared_buffer[threadIdx.x] - coeff * shared_buffer[threadIdx.x-2]);
			break;
	}
	 
	//il pixel rappresentante la trasformata, viene assegnato sull'array dell'immagine trasformata, nella posizione che deve avere secondo la classica allocazione per righe (quindi non secondo la disposizione per la lettura coalescente)
	//si noti che lo stride delle colonne è quello sempre dell'immagine originale
	trasformataDataD[bigI * realN + iSottogruppo * (m/2)*realN + bigJ + jSottogruppo * (n/2)]=val; // calcoliamo l'indice, relativo all'immagine finale (nel sottoquadrante su cui stiamo lavorando) del pixel risultato
	
	if(iSottogruppo == 0 && jSottogruppo == 0) //e soltanto i pixel del sottoquadrante alto a sinistra, del quadrante su cui abbiamo appena calcolato la trasformata, vengono copiati nell'array che deve contenere i pixel da ridisporre al giro successivo (il sottosotto quadrante assegnato conserva la forma classica dell'allocazione per righe di un'immagine)
		nextQuad[bigI * realN + bigJ] = val;
	else nextQuad[bigI * realN + iSottogruppo * (m/2)*realN + bigJ + jSottogruppo * (n/2)] = 0.0f; //altrimenti pongo a 0 l'elemento 
		
	
}


/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*questa routine invece setta la griglia e chiama i kernel (sia quello di ridistribuzione immagine che quello di calcolo trasformata di haar) senza ricopiare i risultati intermedi nella memoria host */
__host__ Mat * callHaarDeviceAllSteps(const Mat & img,int & lvl,float coeff,dim3 sizeGriglia, dim3 sizeBlocco) {
	MioTimer timer; //oggetto per calcolare tempi
	float *imgDataOriginaleH = (float *) img.data;
	float *imgDataOriginaleD;
	float *imgDataRidispostoD;
	float *imgDataTrasformataD;
	float *imgDataTrasformataH;
	int actualLvl = 1;
	int actualM = img.rows;
	int actualN = img.cols;
	
	
	/*parto col calcolo tempi */
	timer.start();
	
	cudaMalloc(&imgDataOriginaleD,img.rows*img.cols*sizeof(float));
	cudaMemset(imgDataOriginaleD,0,img.rows*img.cols*sizeof(float));
	checkErroreCuda("cudaMalloc imgDataOriginaleD");
	cudaMemcpy(imgDataOriginaleD,imgDataOriginaleH,img.rows*img.cols*sizeof(float),cudaMemcpyHostToDevice);
	checkErroreCuda("cudaMemcpy imgDataOriginaleD");
	
	cudaMalloc(&imgDataRidispostoD,img.rows*img.cols*sizeof(float));
	cudaMemset(imgDataRidispostoD,0,img.rows*img.cols*sizeof(float));
	checkErroreCuda("cudaMalloc imgDataRidispostoD");
	
	
	cudaMalloc(&imgDataTrasformataD,img.rows*img.cols*sizeof(float));
	cudaMemset(imgDataTrasformataD,0,img.rows*img.cols*sizeof(float));
	checkErroreCuda("cudaMalloc imgDataTrasformataD");
	
	imgDataTrasformataH = (float *)malloc(img.rows*img.cols*sizeof(float));
	
	
	
	while(actualLvl <= lvl){
			
			if(actualM % 2 != 0 || actualN % 2 != 0 || (actualM * actualN ) % 4 != 0){
				cout<<"nesting interrotto tentando col livello "<<actualLvl<<endl;
				break;
			}
			
			ridisponiPorzione<<<sizeGriglia,sizeBlocco>>>(imgDataOriginaleD,imgDataRidispostoD,actualM,actualN,img.rows,img.cols);
			checkErroreCuda("kernel ridisponiPorzione");
			cudaThreadSynchronize();
			//arrivati a questo punto avremo i pixel del quadrante DI imgDataOriginale che ci interessa, ridisposti in imgDataRidisposto
			trasformataHaarGPUv2<<<sizeGriglia,sizeBlocco,sizeBlocco.x * sizeof(float)>>>(imgDataRidispostoD,imgDataTrasformataD,imgDataOriginaleD,actualM,actualN,img.rows,img.cols,coeff);
			//a questo punto in imgDataTrasformataD c'e' la trasformata fino al livello attuale, ed in imgDataOriginale ci sono i pixel (del sottoquadrante di imgDataTrasformata per il giro successivo)
			//che andranno ridisposti (al giro successivo)
			
			cudaThreadSynchronize();
		
			actualLvl++;
			actualM = actualM / 2;
			actualN = actualN / 2;
	}
	
	if(actualLvl <= lvl) //allora abbiamo interrotto il nesting prima di raggiungere il livello richiesto
		lvl = actualLvl-1; //in tal caso salviamo il livello raggiunto (-1 poichè actualLvl è pre incrementato prima del controllo successivo)
	
	/*stop calcolo tempi ---------------------------------------*/
	cout<<"tempo: livello trasformata: "<<lvl<<" usando GPU , numero threads per blocco: "<<sizeBlocco.x<<" senza copie intermedie : "<<timer.stopECalcolaTempo()<<" ms"<<endl;
	
	
	cudaMemcpy(imgDataTrasformataH,imgDataTrasformataD,img.rows * img.cols * sizeof(float),cudaMemcpyDeviceToHost);
	checkErroreCuda("cudaMemcpy imgDataTrasformataD to Host");
	
	Mat *result = new Mat(img.rows,img.cols,CV_32FC1,imgDataTrasformataH);
	
	cudaFree(imgDataOriginaleD);
	cudaFree(imgDataRidispostoD);
	cudaFree(imgDataTrasformataD);
	
	return result;
	
	
	
}


int main(int argc,char *argv[]){
	
	if(argc!=5){
		cout<<"uso scorretto: [1]path immagine [2]livello ricorsione [3]numero thread per blocco [4]flag mostra immagine"<<endl;
		return 1;
	}
	
	int lvl;
	dim3 sizeBlocco,sizeGriglia; //i blocchi di thread sono dimensionali, così come la griglia. Il numero di thread per blocco deve essere divisibile interamente per 4, e deve essere >=4
	sscanf(argv[2],"%d",&lvl);
	sscanf(argv[3],"%d",&sizeBlocco.x);
	sscanf(argv[4],"%d",&mostraImmagineFlag);
	
	
	Mat imgC = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE); //leggo immagine come grayscale (quindi un unico canale). Automaticamente sarà di tipo CV_8UC1
	if(!imgC.data){
		cout<<"impossibile aprire immagine"<<endl;
		return 1;
	}
	
	if(imgC.cols % 2 != 0  || imgC.cols % 2 != 0 ){ //deve essere per almeno un livello divisibile per 2, altrimenti ne arrotondo le dimensioni in modo tale che lo siano
		resize(imgC, imgC, Size(2*(int)(imgC.cols / 2),2*(int)(imgC.rows / 2)), 0, 0, CV_INTER_LINEAR);
	}
	
	
	if(sizeBlocco.x % 4 != 0 || sizeBlocco.x < 4){
		cout<<"il numero di thread per blocco deve essere divisibile per 4, e >=4"<<endl;
		return 1;
	}
	//per immagine MxN avrò M/2 x N/2 quadruple di pixel, ciascuna che usa 4 threads, quindi in totale MxN threads, divisi in blocchi da sizeBlocco.x thread per blocco.
	//Siccome l'immagine ha le dimensioni che sono multiplo di 2, in totale MxN (che è anche il numero totale di thread oltre che di pixel) è sicuramente multiplo di 4, così come deve esserlo il numero di thread per blocco.
	//Tuttavia può capitare che ci siano gruppi di thread (a 4 a 4) che non lavorano in un blocco.
	sizeGriglia.x = ( (imgC.cols * imgC.rows) % sizeBlocco.x == 0) ? (imgC.cols * imgC.rows) / sizeBlocco.x : 1 + (imgC.cols * imgC.rows) / sizeBlocco.x;
	
	if(sizeGriglia.x > 65535){
		cout<<"scelto un numero di threads per blocco troppo piccolo (rispetto alla dimensione dell'immagine)"<<endl<<"Quindi il numero di blocchi è oltre il limite di 65535!";
		return 1;
	}

	mostraImmagine(imgC,"Immagine Originale");
	
	Mat img(imgC.rows,imgC.cols,CV_32FC1); //siccome lavoro con immagini con pixel in valori float, creo Mat di tipo CV_32FC1
	imgC.convertTo(img,CV_32FC1); //ci converto l'immagine originariamente letta , NB i valori rimangono tra 0 e 255 (sono castati a float, ma non sono normalizzati tra 0 e 1)
	
	cout<<"calcolo trasformata per immagine "<<img.rows<<" x "<<img.cols<<endl;
	
	/*---------------------------approccio su host--------------------------------*/
	
	/* 
	Mat result(img.rows,img.cols,CV_32FC1); //qui verrà salvata la trasformata di Haar, per i livelli scelti. 
	callRecursiveHaarHost(img,result,lvl,sqrt(2.0)/2.0);
	mostraImmagine(result,"Trasforma ta su Host"); */
	 
	
	/*---------------------------approccio su device---------------------------------*/
	//approccio con passi intermedi:
	//Mat *trasformataDevice1 = callRecursiveHaarDeviceCopieIntermedie(img,lvl,sizeBlocco,sizeGriglia,sqrt(2.0)/2.0);
	//mostraImmagine(*trasformataDevice1,"Trasformata su Device con passi multipli "); 
	
	
	//approccio senza copie intermedie tra memoria host e device (e viceversa)
	
	 Mat *trasformataDevice2 = callHaarDeviceAllSteps(img,lvl,sqrt(2.0)/2.0,sizeGriglia,sizeBlocco);
	mostraImmagine(*trasformataDevice2,"Trasformata su Device senza copie intermedie da device");
	
	//salvo immagine trasformata nella cartella output/
	/*stringstream ss;
	ss<<"output/lvl"<<lvl<<"_"<<argv[1];
	salvaImmagine(*trasformataDevice2, ss.str().c_str()); */ 
	
	
	/*---------------------------ricostruzione-----------------------------*/
	
	//Mat reversed1(img.rows,img.cols,CV_32FC1); //qui salvo immagine ricostruita da quella calcolata su host
	//Mat reversed2(img.rows,img.cols,CV_32FC1); //qui salvo immagine ricostruita da quella calcolata con kernel su più passi con copie intermedie
	//Mat reversed3(img.rows,img.cols,CV_32FC1); //qui salvo immagine ricostruita da quella calcolata con kernel senza copie intermedie
	
	//callRecursiveRicostr(result,reversed1,lvl,sqrt(2.0)/2.0); 
	//callRecursiveRicostr(*trasformataDevice1,reversed2,lvl, sqrt(2.0)/2.0); 
	//callRecursiveRicostr(*trasformataDevice2,reversed3,lvl, sqrt(2.0)/2.0);
	
	//mostraImmagine(reversed1,"Ricostruzione da trasformata ottenuta su host");
	//mostraImmagine(reversed2,"Ricostruzione da trasformata con copie da memoria device intermedie");
	//mostraImmagine(reversed3,"Ricostruzione da trasformata senza copie intermedie memorie host-device");
	
	
	
	return 0;
	
	
}