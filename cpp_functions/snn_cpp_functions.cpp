#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>

namespace py = pybind11;

Eigen::SparseMatrix<double> pruneSNN(Eigen::SparseMatrix<double> snnMatrix, double prune = 0) {
	if (prune <= 0) {
		return snnMatrix;
	}

	for (int i = 0; i < snnMatrix.outerSize(); i++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(snnMatrix, i); it; ++it) {
			if (it.value() <= prune) {
				it.valueRef() = 0;
			}
		}
	}

	snnMatrix.prune(0.0); // actually remove pruned values

	return snnMatrix;
}

Eigen::SparseMatrix<double> computeSNN(Eigen::SparseMatrix<double> &nnMatrix, int k, double prune = 0) {
	Eigen::SparseMatrix<double> SNN = nnMatrix * (nnMatrix.transpose());

	for (int i = 0; i < SNN.outerSize(); i++) {
		for (Eigen::SparseMatrix<double>::InnerIterator it(SNN, i); it; ++it) {
			it.valueRef() = it.value()/(k + (k - it.value()));
			if (prune <= 0) {
				continue;
			}

			if (it.value() <= prune) {
				it.valueRef() = 0;
			}
		}
	}

	if (prune > 0) {
 	 	SNN.prune(0.0); // actually remove pruned values
	}

	return(SNN);
}

py::dict getNNmatrix(Eigen::MatrixXi nnRanked, int k = -1, int start = 0, double prune = 0) {
	int nRows = nnRanked.rows(), nCols = nnRanked.cols();

    py::dict result;
    
	if (k == -1 || k > nCols) {
		k = nCols;
	}

	if (start + k > nCols) {
		k = nCols - start;
	}

	// std::cout << nRows << ' ' << nCols << ' ' << k << '\n';
    std::vector<Eigen::Triplet<double>> tripletList;
  	tripletList.reserve(nRows * k);
    
    for (int j = start; j < start + k; j++) {
		for (int i = 0; i < nRows; i++) {
			tripletList.push_back(Eigen::Triplet<double>(i, nnRanked(i, j) - 1, 1));
        }
    }

    Eigen::SparseMatrix<double> NN(nRows, nRows);
	NN.setFromTriplets(tripletList.begin(), tripletList.end());

	if (prune < 0) {
        result["nn"] = NN;
		return result;
	}
	
    
    result["nn"] = NN;
    result["snn"] = computeSNN(NN, k, prune);
    
    return result;
}


PYBIND11_MODULE(snn_functions, m) {
    m.def("computeSNN", &computeSNN, "A function that computes the SNN matrix");
	m.def("pruneSNN", &pruneSNN, "A function that prunes the SNN matrix");
	m.def("getNNmatrix", &getNNmatrix, "A function that computes the SNN matrix");
}

