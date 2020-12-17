#include <cmath>
#include <stdio.h>
#include <random>
#include "mpi.h"
using namespace std;

int ProcCount = 0;
int Rank = 0;
float* A = 0, * B = 0, * C = 0, * A1 = 0, * B1 = 0, * C1 = 0, * T1 = 0;
int MatrSize = 0, BlockSize = 0, GridSize = 0;
int GridCoords[2];
MPI_Comm GridComm, ColComm, RowComm;
MPI_Datatype MPI_BLOCK;

random_device r;
// Choose a random mean between 1 and 6
mt19937 engine(r());
uniform_real_distribution<float> uniform_dist(1, 16);

void Show(float x) {
	printf("%f ", x);
}
void CreateGridComm()
{
	int DimSize[2] = { GridSize, GridSize };
	int Periodic[2] = { 0,0 };
	int SubDims[2];
	MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 0, &GridComm);
	MPI_Cart_coords(GridComm, Rank, 2, GridCoords);
	SubDims[0] = 0;
	SubDims[1] = 1;
	MPI_Cart_sub(GridComm, SubDims, &RowComm);
	SubDims[0] = 1;
	SubDims[1] = 0;
	MPI_Cart_sub(GridComm, SubDims, &ColComm);
}
void InitMatrices()
{
	if (Rank == 0)
		MatrSize = 6; 
	MPI_Bcast(&MatrSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(GridComm);
	BlockSize = MatrSize / GridSize;
	A1 = new float[BlockSize * BlockSize];
	B1 = new float[BlockSize * BlockSize];
	C1 = new float[BlockSize * BlockSize];
	for (int i = 0; i < BlockSize * BlockSize; i++)
		C1[i] = 0;
	if (Rank == 0)
	{
		
		A = new float[MatrSize * MatrSize];
		B = new float[MatrSize * MatrSize];
		C = new float[MatrSize * MatrSize];
		for (int i = 0; i < MatrSize; i++)
			for (int j = 0; j < MatrSize; j++)
			{
				A[i * MatrSize + j] = uniform_dist(engine);
				B[i * MatrSize + j] = uniform_dist(engine);
			}
		//printf("Matrix A\n");
		//for (int i = 0; i < MatrSize; i++)
		//{
		//	for (int j = 0; j < MatrSize; j++)
		//		Show(A[i * MatrSize + j]);
		//	printf("\n");;
		//}
		//printf("Matrix B\n");
		//for (int i = 0; i < MatrSize; i++)
		//{
		//	for (int j = 0; j < MatrSize; j++)
		//		Show(B[i * MatrSize + j]);
		//	printf("\n");
		//}
	}
}
void DataDistribution()
{
	MPI_Type_vector(BlockSize, BlockSize, MatrSize, MPI_FLOAT, &MPI_BLOCK);
	MPI_Type_commit(&MPI_BLOCK);
	if (Rank == 0) {
		for (int r = ProcCount - 1; r > 0; r--)
		{
			int c[2];
			MPI_Cart_coords(GridComm, r, 2, c);
			MPI_Send(A + c[0] * MatrSize * BlockSize + c[1] * BlockSize, 1,
				MPI_BLOCK, r, 0, GridComm);
			MPI_Send(B + c[0] * MatrSize * BlockSize + c[1] * BlockSize, 1,
				MPI_BLOCK, r, 1, GridComm);
		}
		MPI_Request q;
		MPI_Status s;
		int c[2];
		MPI_Cart_coords(GridComm, 0, 2, c);
		MPI_Irecv(A1, BlockSize * BlockSize, MPI_FLOAT, 0, 2, GridComm, &q);
		MPI_Send(A + c[0] * MatrSize * BlockSize + c[1] * BlockSize, 1, MPI_BLOCK, 0, 2, GridComm);		
		MPI_Wait(&q, &s);

		MPI_Irecv(B1, BlockSize * BlockSize, MPI_FLOAT, 0, 1, GridComm, &q);
		MPI_Send(B + c[0] * MatrSize * BlockSize + c[1] * BlockSize, 1, MPI_BLOCK, 0, 1, GridComm);
		MPI_Wait(&q, &s);
	}
	else {
		MPI_Status s;
		MPI_Recv(A1, BlockSize * BlockSize, MPI_FLOAT, 0, 0, GridComm, &s);
		MPI_Recv(B1, BlockSize * BlockSize, MPI_FLOAT, 0, 1, GridComm, &s);
	}

	//printf("Block A\n");
	//for (int i = 0; i < BlockSize; i++)
	//{
	//	for (int j = 0; j < BlockSize; j++)
	//		Show(A1[i * BlockSize + j]);
	//	printf("\n");
	//}
	//printf("Block B\n");
	//for (int i = 0; i < BlockSize; i++)
	//{
	//	for (int j = 0; j < BlockSize; j++)
	//		Show(B1[i * BlockSize + j]);
	//	printf("\n");
	//}
}
void initialRowShift() {	
	int coord[2];
	MPI_Cart_coords(GridComm, Rank, 2, coord);
	MPI_Status s;
	int i = coord[0];
	int j = coord[1];
	int targetJ = (GridSize + (j - i)) % GridSize;
	coord[1] = targetJ;
	int to, from;
	MPI_Cart_rank(GridComm, coord, &to);
	targetJ = (GridSize + (j + i)) % GridSize;
	coord[1] = targetJ;
	MPI_Cart_rank(GridComm, coord, &from);
	MPI_Sendrecv_replace(A1, BlockSize * BlockSize, MPI_FLOAT, to, 0,
		from, 0, GridComm, &s);
}
void CyclicRowShift() {
	int coord[2];
	MPI_Cart_coords(GridComm, Rank, 2, coord);
	MPI_Status s;
	int i = coord[0];
	int j = coord[1];
	int targetJ = (GridSize + j - 1) % GridSize;
	coord[1] = targetJ;
	int to, from;
	MPI_Cart_rank(GridComm, coord, &to);
	targetJ = (GridSize + (j + 1)) % GridSize;
	coord[1] = targetJ;
	MPI_Cart_rank(GridComm, coord, &from);
	MPI_Sendrecv_replace(A1, BlockSize * BlockSize, MPI_FLOAT, to, 0,
		from, 0, GridComm, &s);
}
void initialColumnShift() {
	int coord[2];
	MPI_Cart_coords(GridComm, Rank, 2, coord);
	MPI_Status s;
	int i = coord[0];
	int j = coord[1];
	int targetI = (GridSize + (i - j)) % GridSize;
	coord[0] = targetI;
	int to, from;
	MPI_Cart_rank(GridComm, coord, &to);
	targetI = (GridSize + (i + j)) % GridSize;
	coord[0] = targetI;
	MPI_Cart_rank(GridComm, coord, &from);
	MPI_Sendrecv_replace(B1, BlockSize * BlockSize, MPI_FLOAT, to, 0,
		from, 0, GridComm, &s);
}
void CyclicColumnShift() {
	int coord[2];
	MPI_Cart_coords(GridComm, Rank, 2, coord);
	MPI_Status s;
	int i = coord[0];
	int j = coord[1];
	int targetI = (GridSize + (i - 1)) % GridSize;
	coord[0] = targetI;
	int to, from;
	MPI_Cart_rank(GridComm, coord, &to);
	targetI = (GridSize + (i + 1)) % GridSize;
	coord[0] = targetI;
	MPI_Cart_rank(GridComm, coord, &from);
	MPI_Sendrecv_replace(B1, BlockSize * BlockSize, MPI_FLOAT, to, 0,
		from, 0, GridComm, &s);
}
void A1B1Mult()
{
	for (int i = 0; i < BlockSize; i++)
		for (int j = 0; j < BlockSize; j++)
		{
			double t = 0;
			for (int k = 0; k < BlockSize; k++)
				t += A1[i * BlockSize + k] * B1[k * BlockSize + j];
			C1[i * BlockSize + j] += t;
		}
}
void ParallelCalc()
{
	initialRowShift();
	initialColumnShift();
	for (int i = 0; i < GridSize; i++)
	{
		A1B1Mult();
		CyclicRowShift();
		CyclicColumnShift();
	}
}
void GatherResult()
{
	//printf("Block C\n");
	//for (int i = 0; i < BlockSize; i++)
	//{
	//	for (int j = 0; j < BlockSize; j++)
	//		Show(C1[i * BlockSize + j]);
	//	printf("\n");
	//}
	if (Rank != 0) {
		MPI_Send(C1, BlockSize * BlockSize, MPI_FLOAT, 0, 0, GridComm);
	}
	else
	{
		MPI_Status s;
		MPI_Request q;
		int c[2];
		for (int r = 1; r < ProcCount; r++)
		{
			
			MPI_Cart_coords(GridComm, r, 2, c);
			MPI_Recv(C + c[0] * MatrSize * BlockSize + c[1] * BlockSize, 1,
				MPI_BLOCK, r, 0, GridComm, &s);
		}
		
		MPI_Cart_coords(GridComm, 0, 2, c);
		MPI_Irecv(C + c[0] * MatrSize * BlockSize + c[1] * BlockSize, 1, MPI_BLOCK, 0, 0, GridComm, &q);
		MPI_Send(C1, BlockSize * BlockSize, MPI_FLOAT, 0, 0, GridComm);
		MPI_Wait(&q, &s);


		printf("Matrix C\n");
		for (int i = 0; i < MatrSize; i++)
		{
			for (int j = 0; j < MatrSize; j++)
				Show(C[i * MatrSize + j]);
			printf("\n");
		}
	}
}
void TestResult()
{
	if (Rank == 0)
	{
		printf("Check\n");
		for (int i = 0; i < MatrSize; i++)
		{
			for (int j = 0; j < MatrSize; j++)
			{
				float t = 0;
				for (int k = 0; k < MatrSize; k++)
					t += A[i * MatrSize + k] * B[k * MatrSize + j];
				//Show(C[i * MatrSize + j] - t);
				Show(t);
				printf("\n");
			}
			printf("\n");
		}

	}
}
void Solve()
{
	MPI_Init(NULL, NULL);
	int flag;
	MPI_Initialized(&flag);
	if (flag == 0)
		return;
	MPI_Comm_size(MPI_COMM_WORLD, &ProcCount);
	printf("number of processes: %i", ProcCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
	float* A = 0, * B = 0, * C = 0, * A1 = 0, * B1 = 0, * C1 = 0, * T1 = 0;
	int Size = 0, BlockSize = 0;
	GridSize = (int)sqrt((double)ProcCount);
	if (ProcCount != GridSize * GridSize)
		if (Rank == 0)
		{
			printf("Number of processe is not a full square");
			return;
		}
	CreateGridComm();
	InitMatrices();
	DataDistribution();
	ParallelCalc();
	GatherResult();
	TestResult();
	MPI_Finalize();
}

int main(int* argv, char** argc) {
	double starttime, endtime;
	starttime = MPI_Wtime();
	Solve();
	endtime = MPI_Wtime();
	printf("That took %f seconds\n", endtime - starttime);
	
}
