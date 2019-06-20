package gr.aueb.cs.Distributed_Systems2018;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;


public class Worker {

	public static final String ID = "Worker";

	RealMatrix P;
	RealMatrix C;

	RealMatrix X;
	RealMatrix Y;

	double RAM;
	double CPU;

	Socket workerSocket;
	ObjectInputStream in;
	ObjectOutputStream out;

	public Worker() {
		this.workerSocket = null;
		this.in = null;
		this.out = null;
		this.P = null;
		this.C = null;
		this.X = null;
		this.Y = null;
	}

	public static void main(String[] args) {
		Worker worker = new Worker();
		worker.startWorker();
	}

	public void startWorker() {

		try {
			// connect with Master
			workerSocket = new Socket(Constants.MASTER_IP, Constants.MASTER_PORT);
			out = new ObjectOutputStream(workerSocket.getOutputStream());
			in = new ObjectInputStream(workerSocket.getInputStream());

			String message = (String) in.readObject();
			System.out.println("Server says: " + message);

			out.writeObject(ID);
			out.flush();

			this.P = (RealMatrix) in.readObject();
			System.out.println("Worker received P matrix");

			this.C = (RealMatrix) in.readObject();
			System.out.println("Worker received C matrix");

			this.X = (RealMatrix) in.readUnshared();
			System.out.println("Worker received initial X matrix with dimensions: Raw= " + X.getRowDimension()
					+ " Column= " + X.getColumnDimension());

			this.Y = (RealMatrix) in.readUnshared();
			System.out.println("Worker received initial Y matrix with dimensions: Raw= " + Y.getRowDimension()
					+ " Column= " + Y.getColumnDimension());

			boolean training = true;

			while (training) {

				System.out.println("Worker waits to receive data for training.... ");
				System.out.println("---------");

				// READ WHICH ROWS TO TRAIN AND GET Y MATRIX
				int X_startRaw = (Integer) in.readUnshared();
				int X_endRaw = (Integer) in.readUnshared();
				this.Y = (RealMatrix) in.readUnshared();

				System.out.println("---------");

				// TRAIN X
				RealMatrix X_submatrix_trained = train_X_submatrix(X_startRaw, X_endRaw);
				System.out.println("X_submatrix training finished.... ");
				System.out.println("-------------------------");

				out.writeInt(X_startRaw);
				out.flush();

				out.writeInt(X_endRaw);
				out.flush();

				out.reset();

				out.writeUnshared(X_submatrix_trained);
				out.flush();

				out.writeObject("finished_X_part");
				out.flush();

				// READ WHICH ROWS TO TRAIN AND GET X MATRIX
				int Y_startRaw = (Integer) in.readUnshared();
				int Y_endRaw = (Integer) in.readUnshared();

				this.X = (RealMatrix) in.readUnshared();

				// System.out.println("Received X 557 is " +
				// this.X.getRowMatrix(557).toString());

				// TRAIN Y
				RealMatrix Y_submatrix_trained = train_Y_submatrix(Y_startRaw, Y_endRaw);
				System.out.println("Y_submatrix training finished.... ");
				System.out.println("----");

				out.writeInt(Y_startRaw);
				out.flush();

				out.writeInt(Y_endRaw);
				out.flush();

				out.reset();

				out.writeUnshared(Y_submatrix_trained);
				out.flush();

				out.writeObject("finished_Y_part");
				out.flush();

			}

		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} finally {
			try {
				in.close();
				out.close();
				workerSocket.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public RealMatrix train_Y_submatrix(int Y_startRaw, int Y_endtRaw) {

		// TRAIN Y
		RealMatrix XTX = precompute_XX();
		RealMatrix XT = this.X.transpose();
		RealMatrix I_m = MatrixUtils.createRealIdentityMatrix(Constants.USERS);
		RealMatrix I_f = MatrixUtils.createRealIdentityMatrix(Constants.f);

		System.out.println("Y precomputations finished.......");

		for (int i = Y_startRaw; i < (Y_endtRaw + 1); i++) {
			RealMatrix current_Yi = calculate_Yi(i, XTX, XT, I_m, I_f);

			// if (i==0){
			// System.out.println("current_Yi 0 is "+
			// current_Yi.getRowMatrix(i).toString());
			// }

			this.Y.setRowMatrix(i, current_Yi);
		}

		RealMatrix trained_Y_submatrix = this.Y.getSubMatrix(Y_startRaw, Y_endtRaw, 0, (Y.getColumnDimension() - 1));

		return trained_Y_submatrix;
	}

	public RealMatrix train_X_submatrix(int X_startRaw, int X_endRaw) {

		// TRAIN X
		RealMatrix YTY = precompute_YY();
		RealMatrix YT = this.Y.transpose();
		RealMatrix I_n = MatrixUtils.createRealIdentityMatrix(Constants.POIS);
		RealMatrix I_f = MatrixUtils.createRealIdentityMatrix(Constants.f);

		System.out.println("X precomputations finished.......");

		for (int i = X_startRaw; i < (X_endRaw + 1); i++) {
			RealMatrix current_Xu = calculate_Xu(i, YTY, YT, I_n, I_f);
			this.X.setRowMatrix(i, current_Xu);
		}

		RealMatrix trained_X_submatrix = this.X.getSubMatrix(X_startRaw, X_endRaw, 0, (X.getColumnDimension() - 1));
		System.out.println("------------------");

		return trained_X_submatrix;

	}

	public RealMatrix calculate_Xu(int user, RealMatrix YTY, RealMatrix YT, RealMatrix I_n, RealMatrix I_f) {

		RealMatrix Cu = calculate_Cu(user);
		RealMatrix Pu = calculate_Pu(user);

		// // 1st solution
		// RealMatrix inverse = new
		// LUDecomposition(YT.multiply(Cu).multiply(this.Y).add(I_f.scalarMultiply((double)
		// Constants.l))).getSolver().getInverse();
		//
		// RealMatrix Xu = inverse.multiply(YT).multiply(Cu).multiply(Pu);

		// speedup solution
		RealMatrix YT_Cu_Y = YTY.add(YT.multiply(Cu.subtract(I_n)).multiply(this.Y));

		RealMatrix inverse = new LUDecomposition(YT_Cu_Y.add(I_f.scalarMultiply((double) Constants.l))).getSolver()
				.getInverse();

		RealMatrix Xu = inverse.multiply(YT).multiply(Cu).multiply(Pu);

		return Xu.transpose();
	}

	public RealMatrix calculate_Yi(int item, RealMatrix XTX, RealMatrix XT, RealMatrix I_m, RealMatrix I_f) {

		RealMatrix Ci = calculate_Ci(item);
		RealMatrix Pi = calculate_Pi(item);

		// 1st solution
		// RealMatrix inverse = new
		// LUDecomposition(XT.multiply(Ci).multiply(this.X).add(I_f.scalarMultiply((double)
		// Constants.l))).getSolver().getInverse();
		//
		// RealMatrix Yi = inverse.multiply(XT).multiply(Ci).multiply(Pi);

		// speedup solution
		RealMatrix XT_Ci_X = XTX.add(XT.multiply(Ci.subtract(I_m)).multiply(this.X));

		RealMatrix inverse = new LUDecomposition(XT_Ci_X.add(I_f.scalarMultiply((double) Constants.l))).getSolver()
				.getInverse();

		RealMatrix Yi = inverse.multiply(XT).multiply(Ci).multiply(Pi);
		return Yi.transpose();
	}

	public RealMatrix calculate_Cu(int user) {

		// Diagonal Matrix Creation
		double[] diag_elems = this.C.getRow(user);
		RealMatrix Cu = MatrixUtils.createRealDiagonalMatrix(diag_elems);

		return Cu;

	}

	public RealMatrix calculate_Ci(int item) {

		// Diagonal Matrix Creation
		double[] diag_elems = this.C.getColumn(item);
		RealMatrix Ci = MatrixUtils.createRealDiagonalMatrix(diag_elems);

		return Ci;

	}

	public RealMatrix calculate_Pu(int user) {

		RealMatrix Pu = this.P.getRowMatrix(user);

		return Pu.transpose();

	}

	public RealMatrix calculate_Pi(int item) {

		RealMatrix Pi = this.P.getColumnMatrix(item);
		return Pi;

	}

	public RealMatrix precompute_YY() {

		RealMatrix YY = this.Y.transpose().multiply(this.Y);
		return YY;
	}

	public RealMatrix precompute_XX() {

		RealMatrix XX = this.X.transpose().multiply(this.X);
		return XX;
	}

	public RealMatrix getX() {
		return X;
	}

	public void setX(RealMatrix x) {
		X = x;
	}

	public RealMatrix getY() {
		return Y;
	}

	public void setY(RealMatrix y) {
		Y = y;
	}

}
