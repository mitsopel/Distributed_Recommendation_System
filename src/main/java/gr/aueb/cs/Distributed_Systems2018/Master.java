package gr.aueb.cs.Distributed_Systems2018;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

public class Master {

	public static final int PORT = Constants.MASTER_PORT;
	public static final int RAWS = Constants.USERS;
	public static final int COLUMNS = Constants.POIS;

	ServerSocket serverSocket;
	List<Connection> establishedConnections;

	double[][] r;
	double[][] p;
	double[][] c;

	RealMatrix R;
	RealMatrix P;
	RealMatrix C;

	public RealMatrix X;
	public RealMatrix Y;

	public Master() {
		super();
		this.serverSocket = null;
		this.establishedConnections = new ArrayList<Connection>();
		this.r = null;
		this.p = null;
		this.c = null;
		this.R = null;
		this.P = null;
		this.C = null;
		this.X = null;
		this.Y = null;
	}

	public static void main(String[] args) {

		Master master = new Master();
		Utils.init_JDKRandomGenerator();
		master.initializeMatrixes();

		// System.out.println(master.p[0][0]+" "+ master.p[0][149]);
		// System.out.println(master.p[3][558]+" "+ master.p[3][1751]);
		// System.out.println(master.c[0][0]+" "+ master.c[0][149]);
		// System.out.println(master.c[3][558]+" "+ master.c[3][1751]);

		master.startMaster();
	}

	public void initializeMatrixes() {
		this.createR();
		this.createP();
		this.createC();
		this.generateRandom_X();
		this.generateRandom_Y();
	}

	public void startMaster() {

		Socket socket = null;
		ObjectOutputStream out = null;
		ObjectInputStream in = null;
		String connectionType = null;

		try {
			serverSocket = new ServerSocket(PORT);
			System.out.println("Master is up and waiting for connections..!");
			System.out.println("*********************");

			while (true) {
				socket = serverSocket.accept();

				out = new ObjectOutputStream(socket.getOutputStream());
				in = new ObjectInputStream(socket.getInputStream());

				out.writeObject("Connection with Master is established!");

				connectionType = (String) in.readObject();
				System.out.println("-------------------------");
				System.out.println("Master established a connection with a " + connectionType + "  "
						+ socket.getInetAddress().getHostName());

				establishedConnections.add(new Connection(connectionType, socket, out, in));

				if (connectionType.equalsIgnoreCase("Worker")) {

					new Thread(new Runnable() {
						public void run() {

							Connection worker = establishedConnections.get(establishedConnections.size() - 1);

							try {
								worker.out.writeObject(P);
								worker.out.flush();
								System.out.println("Master sent P matrix");

								worker.out.writeObject(C);
								worker.out.flush();
								System.out.println("Master sent C matrix");

								worker.out.writeUnshared(X);
								worker.out.flush();
								System.out.println("Master sent X matrix with dimensions: Raw= " + X.getRowDimension()
										+ " Column= " + X.getColumnDimension());

								worker.out.writeUnshared(Y);
								worker.out.flush();
								System.out.println("Master sent Y matrix with dimensions: Raw= " + Y.getRowDimension()
										+ " Column= " + Y.getColumnDimension());

							} catch (IOException e) {
								e.printStackTrace();
							}
						}
					}).start();
				}

				else if (connectionType.equalsIgnoreCase("Client")) {

					new Thread(new Runnable() {
						public void run() {

							Connection client = establishedConnections.get(establishedConnections.size() - 1);

							System.out.println("--------");

							double error = 0;
							int i = 0;
							while (i < Constants.epochs) {
								i++;

								distribute_Xrows_and_Y_to_workers();

								boolean is_X_finished = is_X_trained();

								if (is_X_finished) {

									distribute_Yrows_and_X_to_workers();

								} else {
									System.out.println("Error at X training....... ERROR!!!");

								}

								boolean is_Y_finished = is_Y_trained();

								if (is_Y_finished) {

									System.out.println("--------------");

									error = calculateError();
									System.out.println("Epoch " + i + " Error is: " + error);

								}
							}

							// CALCULATE USER RECCOMENDATIONS......!!!
							// calculateUserReccomendations(0, 5);

							try {
								client.out.writeObject(error);
							} catch (IOException e) {
								e.printStackTrace();
							}

						}
					}).start();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} finally {
			for (Connection connection : establishedConnections) {
				try {
					connection.socket.close();
					connection.in.close();
					connection.out.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			try {
				serverSocket.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public void calculateUserReccomendations(int user, int topK) {

		List<Poi> reccomendations = new ArrayList<Poi>();

		RealMatrix Xu = this.X.getRowMatrix(user);

		double XuYi_value = 0;

		for (int j = 0; j < Y.getRowDimension(); j++) {

			RealMatrix Yi = Y.getRowMatrix(j).transpose();
			RealMatrix XuYi = Xu.multiply(Yi);
			XuYi_value = XuYi.getEntry(0, 0);

			Poi poi = new Poi(user, j, XuYi_value);

			reccomendations.add(poi);
		}

		Collections.sort(reccomendations, new Comparator<Poi>() {
			public int compare(Poi left, Poi right) {

				return left.valueForUser > right.valueForUser ? -1 : (left.valueForUser < right.valueForUser ? 1 : 0);
			}
		});

		System.out.println("Sorted recommendation is: " + reccomendations.get(0).valueForUser);
		System.out.println("------------");

		System.out.println("Recommendations including already visited");
		for (int i = 0; i < topK; i++) {

			System.out.println("POI number: " + reccomendations.get(i).PoiID);
			System.out.println("POI score: " + reccomendations.get(i).valueForUser);
			System.out.println("  ");
		}
		System.out.println("------------");

		System.out.println("Recommendations NOT including already visited");

		int topK_notVisisted = 0;
		int k = 0;

		while (topK_notVisisted < topK) {

			int currentPOI_ID = reccomendations.get(k).PoiID;

			if (this.P.getEntry(user, currentPOI_ID) == 0) {
				System.out.println("POI number: " + reccomendations.get(k).PoiID);
				System.out.println("POI score: " + reccomendations.get(k).valueForUser);
				System.out.println("  ");

				topK_notVisisted++;
				k++;
			} else {
				k++;
			}
		}
	}

	public synchronized double calculateError() {

		// System.out.println("Calculating Error....");

		double totalError = 0;

		double sum = 0;
		double normalizationFactor = calculateNormalizationFactor();

		double current_Cui = 0;
		double current_Pui = 0;

		double XuYi_value = 0;

		double Pui_XuYi = 0;

		double Cui_Pui_XuYi = 0;

		for (int i = 0; i < X.getRowDimension(); i++) {

			RealMatrix Xu = X.getRowMatrix(i);

			for (int j = 0; j < Y.getRowDimension(); j++) {

				RealMatrix Yi = Y.getRowMatrix(j).transpose();

				current_Cui = C.getEntry(i, j);
				current_Pui = P.getEntry(i, j);

				RealMatrix XuYi = Xu.multiply(Yi);

				XuYi_value = XuYi.getEntry(0, 0);

				Pui_XuYi = Math.pow(current_Pui - XuYi_value, 2);

				Cui_Pui_XuYi = current_Cui * Pui_XuYi;

				sum += Cui_Pui_XuYi;
			}
		}

		totalError = sum + normalizationFactor;
		return totalError;

	}

	public synchronized double calculateNormalizationFactor() {

		double sum_Xu = 0;
		double sum_Yi = 0;

		double current_Xui = 0;
		double squared_current_Xui = 0;

		for (int i = 0; i < X.getRowDimension(); i++) {
			for (int j = 0; j < X.getColumnDimension(); j++) {

				current_Xui = X.getEntry(i, j);

				squared_current_Xui = Math.pow(current_Xui, 2);
				sum_Xu += squared_current_Xui;
			}
		}

		double current_Yui = 0;
		double squared_current_Yui = 0;

		for (int i = 0; i < Y.getRowDimension(); i++) {
			for (int j = 0; j < Y.getColumnDimension(); j++) {
				current_Yui = Y.getEntry(i, j);
				squared_current_Yui = Math.pow(current_Yui, 2);
				sum_Yi += squared_current_Yui;
			}
		}

		double normalizationFactor = Constants.l * (sum_Xu + sum_Yi);

		return normalizationFactor;
	}

	public boolean is_Y_trained() {

		int counter = 0;
		for (Connection connection : establishedConnections) {
			if (connection.name.equalsIgnoreCase("Worker")) {
				try {

					int Y_startRaw = (int) connection.in.readInt();

					int Y_endRaw = (int) connection.in.readInt();

					RealMatrix trained_Y_submatrix = (RealMatrix) connection.in.readUnshared();

					for (int i = 0; i < trained_Y_submatrix.getRowDimension(); i++) {

						RealMatrix currentRow = trained_Y_submatrix.getRowMatrix(i);

						this.Y.setRowMatrix(Y_startRaw, currentRow);

						Y_startRaw++;
					}

					String ACK = (String) connection.in.readObject();

					if (ACK.equalsIgnoreCase("finished_Y_part")) {
						counter++;
					}

				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		if (counter == workersSize()) {
			return true;
		} else {
			System.out.println("Y_is_trained() method at Master did not receive all workers trained data");
			return false;
		}

	}

	public synchronized boolean is_X_trained() {

		int counter = 0;
		for (Connection connection : establishedConnections) {
			if (connection.name.equalsIgnoreCase("Worker")) {
				try {

					int X_startRaw = (int) connection.in.readInt();

					int X_endRaw = (int) connection.in.readInt();

					RealMatrix trained_X_submatrix = (RealMatrix) connection.in.readUnshared();

					for (int i = 0; i < trained_X_submatrix.getRowDimension(); i++) {

						RealMatrix currentRow = trained_X_submatrix.getRowMatrix(i);

						this.X.setRowMatrix(X_startRaw, currentRow);

						X_startRaw++;
					}

					// System.out.println("X 557 is " +
					// this.X.getRowMatrix(557).toString());

					String ACK = (String) connection.in.readObject();

					if (ACK.equalsIgnoreCase("finished_X_part")) {
						counter++;
					}

				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		if (counter == workersSize()) {
			return true;
		} else {
			System.out.println("X_is_trained() method at Master did not receive all workers trained data");
			return false;
		}

	}

	public void distribute_Xrows_and_Y_to_workers() {

		int workersSize = workersSize();

		int X_portion = (int) X.getRowDimension() / workersSize;
		// -1 due to the [0] element of the array
		int X_portion_remaining = (int) ((X.getRowDimension() % workersSize) - 1);

		int current_startRow = 0;
		int current_endRow = (X_portion + X_portion_remaining);

		for (Connection connection : establishedConnections) {

			if (connection.name.equalsIgnoreCase("Worker")) {

				try {
					connection.out.writeUnshared(current_startRow);
					connection.out.flush();

					connection.out.writeUnshared(current_endRow);
					connection.out.flush();

					connection.out.reset();

					connection.out.writeUnshared(this.Y);
					connection.out.flush();

				} catch (OutOfRangeException e) {
					e.printStackTrace();
				} catch (NumberIsTooSmallException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			current_startRow = (current_endRow + 1);
			current_endRow += X_portion;
		}

	}

	public void distribute_Yrows_and_X_to_workers() {

		int workersSize = workersSize();

		int Y_portion = (int) Y.getRowDimension() / workersSize;
		// -1 due to the [0] element of the array
		int Y_portion_remaining = (int) ((Y.getRowDimension() % workersSize) - 1);

		int current_startRow = 0;
		int current_endRow = (Y_portion + Y_portion_remaining);

		// System.out.println("Distribute NEW X to workers");

		for (Connection connection : establishedConnections) {
			if (connection.name.equalsIgnoreCase("Worker")) {

				try {
					connection.out.writeUnshared(current_startRow);
					connection.out.flush();

					connection.out.writeUnshared(current_endRow);
					connection.out.flush();

					// System.out.println("NEW X 557 is " +
					// this.X.getRowMatrix(557).toString());

					connection.out.reset();

					connection.out.writeUnshared(this.X);
					connection.out.flush();

				} catch (OutOfRangeException e) {
					e.printStackTrace();
				} catch (NumberIsTooSmallException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			current_startRow = (current_endRow + 1);
			current_endRow += Y_portion;
		}

	}

	public void createR() {
		r = new double[RAWS][COLUMNS];

		Scanner scanner = null;
		try {
			scanner = new Scanner(new File(
					"C:\\Users\\Ellen\\Documents\\Eclipse\\workspace\\distributed_systems_2018\\src\\Data\\input_matrix_no_zeros.csv"));

			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] values = line.split(",");

				// System.out.println(values[0] + " " + values[1] + " "
				// +values[2]);

				int raw = Integer.parseInt(values[0].replace(" ", ""));
				int column = Integer.parseInt(values[1].replace(" ", ""));
				int value = Integer.parseInt(values[2].replace(" ", ""));

				r[raw][column] = value;
			}

			R = new Array2DRowRealMatrix(r);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			scanner.close();
		}
	}

	public void createP() {
		p = new double[RAWS][COLUMNS];

		for (int i = 0; i < RAWS; i++) {
			for (int j = 0; j < COLUMNS; j++) {
				p[i][j] = (r[i][j] > 0) ? 1 : 0;
			}
		}

		P = new Array2DRowRealMatrix(p);
	}

	public void createC() {
		c = new double[RAWS][COLUMNS];

		for (int i = 0; i < RAWS; i++) {
			for (int j = 0; j < COLUMNS; j++) {
				c[i][j] = 1 + (Constants.a * r[i][j]);
			}
		}
		C = new Array2DRowRealMatrix(c);
	}

	public void generateRandom_X() {
		// Zero Array Creation
		this.X = MatrixUtils.createRealMatrix(Constants.USERS, Constants.f);


		for (int i = 0; i < X.getRowDimension(); i++) {
			for (int j = 0; j < X.getColumnDimension(); j++) {

//				double rndDouble = Utils.generateRandomDouble(0, 1);
				double rndDouble = Utils.randomGenerator.nextDouble();
				this.X.setEntry(i, j, rndDouble);
			}
		}
	}

	public void generateRandom_Y() {
		// Zero Array Creation
		this.Y = MatrixUtils.createRealMatrix(Constants.POIS, Constants.f);


		for (int i = 0; i < Y.getRowDimension(); i++) {
			for (int j = 0; j < Y.getColumnDimension(); j++) {

//				double rndDouble = Utils.generateRandomDouble(0, 1);
				double rndDouble = Utils.randomGenerator.nextDouble();
				this.Y.setEntry(i, j, rndDouble);
			}
		}
	}

	public int workersSize() {
		int workersSize = 0;

		for (Connection connection : establishedConnections) {
			if (connection.name.equalsIgnoreCase("Worker")) {
				workersSize++;
			}
		}
		return workersSize;
	}

}
