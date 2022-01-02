-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jan 02, 2022 at 08:00 AM
-- Server version: 10.4.21-MariaDB
-- PHP Version: 8.0.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `bigproject`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL,
  `nama` varchar(100) NOT NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`id`, `nama`, `username`, `password`) VALUES
(1, 'Admin Tuing', 'admin', 'admin123');

-- --------------------------------------------------------

--
-- Table structure for table `daftartamu`
--

CREATE TABLE `daftartamu` (
  `id` int(11) NOT NULL,
  `nama_lengkap` varchar(50) NOT NULL,
  `Instansi` varchar(30) NOT NULL,
  `no_telp` varchar(14) NOT NULL,
  `keperluan` text NOT NULL,
  `tanggal` date NOT NULL DEFAULT current_timestamp(),
  `waktu` time NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `daftartamu`
--

INSERT INTO `daftartamu` (`id`, `nama_lengkap`, `Instansi`, `no_telp`, `keperluan`, `tanggal`, `waktu`) VALUES
(51, 'adit', 'smk 1 tegal', '08771268745', 'berkunjung', '2022-01-01', '11:40:54'),
(52, 'Harto', '-', '082372635463', 'Melatih pencak Silat', '2022-01-01', '11:54:20'),
(53, 'Nalaratih', 'Radar Tegal', '085263547364', 'meliput Aacara', '2022-01-01', '11:58:16'),
(54, 'Fakhrudin', 'Universitas Gadjah Mada', '085273648576', 'Studi Banding', '2022-01-02', '12:00:10'),
(55, 'Nadeo Argawinata', 'Universitas Indonesia', '085298976534', 'Undangan Organisasi', '2022-01-02', '12:01:45'),
(56, 'Ricki Kambuaya', 'Universitas Cendrawasih', '085223423412', 'Melamar Kerja', '2022-01-02', '12:06:12'),
(57, 'Witan Sulaeman', 'Universitas panca Sakti', '082372737465', 'Studi Banding Organisasi', '2022-01-03', '12:09:11'),
(58, 'Dedik Setiawan', 'Pemerintah Desa Blubuk', '0823542635213', 'Bertemu Dosen', '2022-01-03', '12:12:11'),
(59, 'Elkan Baggot', 'Pemerintah Kabuaten Brebes', '082354256734', 'Bertemu Wadir 3', '2022-01-02', '12:13:00'),
(60, 'Asnawi Mangkualam', '-', '082372637485', 'Melatih Seakbola', '2022-01-02', '12:13:36');

-- --------------------------------------------------------

--
-- Table structure for table `karyawan`
--

CREATE TABLE `karyawan` (
  `id` int(11) NOT NULL,
  `nama_lengkap` varchar(35) NOT NULL,
  `email` varchar(35) NOT NULL,
  `no_telp` varchar(14) NOT NULL,
  `alamat` varchar(70) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `karyawan`
--

INSERT INTO `karyawan` (`id`, `nama_lengkap`, `email`, `no_telp`, `alamat`) VALUES
(1, 'Doni Cahya', 'zizam2340@gmail.com', '082637287818', 'Pangkah'),
(2, 'Arief Rachman', 'ariefeira713@gmail.com', '082135816581', 'mintaragen'),
(3, 'Adi Sangjaya', 'adi123@gmail.com', '082637738318', 'Pangkah'),
(4, 'Efendi Kisnoto', 'efendi456@gmail.com', '085737657818', 'Pagebarang'),
(5, 'Insan Maulana', 'insanmau@gmial.com', '0817825057505', 'balapulang'),
(6, 'Rziki Dwi Saputra', 'rizki123@gmai.com', '085214568765', 'balamoa'),
(7, 'Ikhlasul Amal ', 'amal90@gmail.com', '085712345678', 'adiwerna'),
(8, 'Ariffullah', 'arivf@gmail.com', '087713468976', 'Pemalang'),
(9, 'Faizal Aji Wibowo', 'faizal678@gmail.com', '087712785647', 'Tegal'),
(10, 'Nurlaela', 'ela879@gmail.com', '085723451678', 'Pangkah'),
(11, 'Elang Bimantoro', 'elang22@gmail.com', '082345781345', 'Brebes'),
(12, 'Irvan Akbar', 'irvan66@gmail.com', '087865432787', 'Dukuh Turi'),
(13, 'Ade Noval', 'mnoval@gmail.com', '08567825617', 'Lebaksiu'),
(14, 'Musnadil Firdaus', 'nadil88@gmail.com', '082356741267', 'Pagongan'),
(15, 'Muhammad Rizky', 'rizky77@gmail.com', '082387651987', 'Brebes'),
(16, 'Ilham Maulana Fajar Sidik', 'ilhamf@gmail.com', '082356128724', 'Randugunting'),
(17, 'Farid Nurul Ihsani', 'farid676@gmail.com', '08237643176', 'Dukuhwaru'),
(18, 'Rizki Fauzi Maksum', 'maksum44@gmail.com', '087776349987', 'Tegal Selatan');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `daftartamu`
--
ALTER TABLE `daftartamu`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `karyawan`
--
ALTER TABLE `karyawan`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `daftartamu`
--
ALTER TABLE `daftartamu`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=61;

--
-- AUTO_INCREMENT for table `karyawan`
--
ALTER TABLE `karyawan`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=19;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
