    def prepare_bundle_adjust_param(self, camera_id_list=None, max_num_images=1000):
        ignore_joint_list = config["skeleton"].ignore_joint_id
        # logger.debug("Calibration ignore joint list {}".format(ignore_joint_list))
        if camera_id_list is None:
            camera_id_list = list(range(self.num_cameras))

        camera_params = np.zeros(shape=(len(self.cam_list), 13), dtype=float)
        cam_list = self.cam_list  # [self.cam_list[c] for c in camera_id_list]
        for cid in range(len(self.cam_list)):
            camera_params[cid, 0:3] = np.squeeze(cam_list[cid].rvec)
            camera_params[cid, 3:6] = np.squeeze(cam_list[cid].tvec)
            camera_params[cid, 6] = cam_list[cid].focal_length_x
            camera_params[cid, 7] = cam_list[cid].focal_length_y
            camera_params[cid, 8:13] = np.squeeze(cam_list[cid].distort)

        point_indices = []
        camera_indices = []
        points2d_ba = []
        points3d_ba = []
        # points3d_ba_source = dict()
        # points3d_ba_source_inv = dict()
        point_index_counter = 0
        s = self.points3d.shape

        img_id_list = np.arange(s[0] - 1)
        if s[0] > max_num_images:
            logger.debug(
                "There are too many ({}) images for calibration. Selecting {} randomly.".format(
                    s[0], max_num_images
                )
            )
            img_id_list = np.random.randint(0, high=s[0] - 1, size=(max_num_images))

        for img_id, j_id in product(img_id_list, range(s[1])):
            # cam_list_iter = list()
            # points2d_iter = list()
            for cam_idx, cam in enumerate(cam_list):
                if (
                    j_id not in ignore_joint_list
                    and not np.any(self.points3d[img_id, j_id, :] == 0)
                    and not np.any(cam[img_id, j_id, :] == 0)
                    and config["skeleton"].camera_see_joint(cam.cam_id, j_id)
                    # and cam.cam_id != 3
                    and cam_idx in camera_id_list
                ):
                    points2d_ba.extend(cam[img_id, j_id, :])
                    points3d_ba.append(self.points3d[img_id, j_id, :])
                    point_indices.append(point_index_counter)
                    point_index_counter += 1
                    camera_indices.append(cam_idx)

                # cam_list_iter.append(cam)
                # points2d_iter.append(cam[img_id, j_id, :])

                # the point is seen by at least two cameras, add it to the bundle adjustment
                """
                if len(cam_list_iter) >= 2:
                    points3d_iter = self.points3d[img_id, j_id, :]
                    points2d_ba.extend(points2d_iter)
                    points3d_ba.append(points3d_iter)
                    point_indices.extend([point_index_counter] * len(cam_list_iter))
                    #points3d_ba_source[(img_id, j_id)] = point_index_counter
                    #points3d_ba_source_inv[point_index_counter] = (img_id, j_id)
                    point_index_counter += 1
                    camera_indices.extend([cam.cam_id for cam in cam_list_iter])
                """
        # c = 0

        # make sure stripes from both sides share the same point id's
        # TODO move this into config file
        """
        if "fly" in config["name"]:
            for idx, point_idx in enumerate(point_indices):
                img_id, j_id = points3d_ba_source_inv[point_idx]
                if (
                    config["skeleton"].is_tracked_point(
                        j_id, config["skeleton"].Tracked.STRIPE
                    )
                    and j_id > config["skeleton"].num_joints // 2
                ):
                    if (
                        img_id,
                        j_id - config["skeleton"].num_joints // 2,
                    ) in points3d_ba_source:
                        point_indices[idx] = points3d_ba_source[
                            (img_id, j_id - config["skeleton"].num_joints // 2)
                        ]
                        c += 1
        """
        # logger.debug("Replaced {} points".format(c))
        points3d_ba = np.squeeze(np.array(points3d_ba))
        points2d_ba = np.squeeze(np.array(points2d_ba))
        # print(points3d_ba.shape, points2d_ba.s, s,)
        # cid2cidx = {v: k for (k, v) in enumerate(np.sort(np.unique(camera_indices)))}
        # camera_indices = [cid2cidx[cid] for cid in camera_indices]
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)

        n_cameras = camera_params.shape[0]
        n_points = points3d_ba.shape[0]

        x0 = np.hstack((camera_params.ravel(), points3d_ba.ravel()))

        return (
            x0.copy(),
            points2d_ba.copy(),
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
        )

    def calibrate(self, cam_id_list=None):
        assert self.cam_list
        ignore_joint_list = config["skeleton"].ignore_joint_id
        if cam_id_list is None:
            cam_id_list = range(self.num_cameras)

        self.reprojection_error()
        (
            x0,
            points_2d,
            n_cameras,
            n_points,
            camera_indices,
            point_indices,
        ) = self.prepare_bundle_adjust_param(cam_id_list)
        logger.debug(f"Number of points for calibration: {n_points}")
        A = bundle_adjustment_sparsity(
            n_cameras, n_points, camera_indices, point_indices
        )
        res = least_squares(
            residuals,
            x0,
            jac_sparsity=A,
            verbose=2 if logger.debug_enabled() else 0,
            x_scale="jac",
            ftol=1e-4,
            method="trf",
            args=(
                self.cam_list,
                n_cameras,
                n_points,
                camera_indices,
                point_indices,
                points_2d,
            ),
            max_nfev=1000,
        )

        logger.debug(
            "Bundle adjustment, Average reprojection error: {}".format(
                np.mean(np.abs(res.fun))
            )
        )

        self.triangulate()
        return res