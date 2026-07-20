"""
鑲岃倝鐗瑰緛璁＄畻鍣?瀹炵幇 2.1 鍩虹褰㈡€佸鐗瑰緛銆?.2 鐏板害/淇″彿鐗瑰緛銆?.3 绌洪棿鍒嗗竷鐗瑰緛銆?.4 绾圭悊鐗瑰緛
"""

import numpy as np
from scipy import ndimage, stats
from skimage import measure, morphology
import cv2
from fat_threshold_v7 import get_fat_threshold_per_muscle
try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    print('Warning: PyRadiomics not found. Texture features will be skipped.')


def calculate_morphological_features(
    muscle_mask: np.ndarray,
    pixel_spacing: tuple
) -> dict:
    """
    璁＄畻 2.1 鍩虹褰㈡€佸鐗瑰緛锛?3椤癸級
    
    Args:
        muscle_mask: 鑲岃倝浜屽€兼帺鐮?(H, W)
        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛 (dx, dy)锛屽崟浣?mm
    
    Returns:
        褰㈡€佸鐗瑰緛瀛楀吀
    """
    features = {}
    dx, dy = pixel_spacing
    
    if not np.any(muscle_mask):
        features = {
            'Area': 0.0,
            'Perimeter': 0.0,
            'Equivalent_Diameter': 0.0,
            'Aspect_Ratio': 0.0,
            'Max_Transverse_Diameter': 0.0,
            'Max_AP_Diameter': 0.0,
            'Circularity': 0.0,
            'Eccentricity': 0.0,
            'Convex_Hull_Area': 0.0,
            'Solidity': 0.0,
            'Max_Inscribed_Circle_Diameter': 0.0,
            'Min_BBox_Orientation': 0.0,
            'Shape_Complexity': 0.0
        }
        return features
    
    contours, _ = cv2.findContours(
        muscle_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        features.update({
            'Area': 0.0,
            'Perimeter': 0.0,
            'Equivalent_Diameter': 0.0,
            'Aspect_Ratio': 0.0,
            'Max_Transverse_Diameter': 0.0,
            'Max_AP_Diameter': 0.0,
            'Circularity': 0.0,
            'Eccentricity': 0.0,
            'Convex_Hull_Area': 0.0,
            'Solidity': 0.0,
            'Max_Inscribed_Circle_Diameter': 0.0,
            'Min_BBox_Orientation': 0.0,
            'Shape_Complexity': 0.0
        })
        return features
    
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour[:, 0, :]
    
    pixel_area = dx * dy
    area = np.sum(muscle_mask) * pixel_area
    features['Area'] = area
    
    if len(contour_points) >= 2:
        perimeter = 0.0
        for i in range(len(contour_points)):
            x1, y1 = contour_points[i]
            x2, y2 = contour_points[(i + 1) % len(contour_points)]
            dx_pixel = (x2 - x1) * dx
            dy_pixel = (y2 - y1) * dy
            perimeter += np.sqrt(dx_pixel**2 + dy_pixel**2)
        features['Perimeter'] = perimeter
    else:
        features['Perimeter'] = 0.0
    
    features['Equivalent_Diameter'] = np.sqrt(4 * area / np.pi) if area > 0 else 0.0
    
    if len(contour_points) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour_points.astype(np.float32))
            major_axis = ellipse[1][0] * dx
            minor_axis = ellipse[1][1] * dy
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0
            features['Aspect_Ratio'] = aspect_ratio
            
            eccentricity = np.sqrt(1 - (min(major_axis, minor_axis) / max(major_axis, minor_axis))**2) if max(major_axis, minor_axis) > 0 else 0.0
            features['Eccentricity'] = eccentricity
            features['Min_BBox_Orientation'] = ellipse[2]
        except:
            features['Aspect_Ratio'] = 0.0
            features['Eccentricity'] = 0.0
            features['Min_BBox_Orientation'] = 0.0
    else:
        features['Aspect_Ratio'] = 0.0
        features['Eccentricity'] = 0.0
        features['Min_BBox_Orientation'] = 0.0
    
    y_coords, x_coords = np.where(muscle_mask)
    max_transverse_diameter = (np.max(x_coords) - np.min(x_coords)) * dx
    features['Max_Transverse_Diameter'] = max_transverse_diameter
    
    max_ap_diameter = (np.max(y_coords) - np.min(y_coords)) * dy
    features['Max_AP_Diameter'] = max_ap_diameter
    
    features['Equivalent_Diameter'] = np.sqrt(4 * area / np.pi) if area > 0 else 0.0
    
    if features['Perimeter'] > 0:
        circularity = (4 * np.pi * area) / (features['Perimeter'] ** 2)
        features['Circularity'] = min(circularity, 1.0)
    else:
        features['Circularity'] = 0.0
    
    try:
        # 分子和分母统一采用离散像素面积。原实现的分子包含全部连通域，
        # 分母却只计算最大轮廓的凸包，因此会产生 Solidity > 1。
        convex_mask = morphology.convex_hull_image(muscle_mask.astype(bool))
        convex_hull_area_mm2 = float(np.sum(convex_mask)) * pixel_area
        features['Convex_Hull_Area'] = convex_hull_area_mm2
        
        if convex_hull_area_mm2 > 0:
            features['Solidity'] = features['Area'] / convex_hull_area_mm2
        else:
            features['Solidity'] = 0.0
    except:
        features['Convex_Hull_Area'] = 0.0
        features['Solidity'] = 0.0
    
    try:
        distance_transform = ndimage.distance_transform_edt(muscle_mask.astype(np.int32))
        max_distance_pixels = np.max(distance_transform)
        max_distance_mm = max_distance_pixels * np.mean(pixel_spacing)
        features['Max_Inscribed_Circle_Diameter'] = max_distance_mm * 2
    except:
        features['Max_Inscribed_Circle_Diameter'] = 0.0
    
    if features['Area'] > 0:
        shape_complexity = (features['Perimeter'] ** 2) / (4 * np.pi * features['Area'])
        features['Shape_Complexity'] = shape_complexity
    else:
        features['Shape_Complexity'] = 0.0
    
    return features


def calculate_spatial_features(
    muscle_mask: np.ndarray,
    fat_mask: np.ndarray,
    pixel_spacing: tuple
) -> dict:
    """
    璁＄畻 2.3 绌洪棿鍒嗗竷鐗瑰緛锛?椤癸級
    
    Args:
        muscle_mask: 鑲岃倝浜屽€兼帺鐮?(H, W)
        fat_mask: 鑴傝偑浜屽€兼帺鐮?(H, W)
        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛 (dx, dy)
    
    Returns:
        绌洪棿鍒嗗竷鐗瑰緛瀛楀吀
    """
    features = {}
    dx, dy = pixel_spacing
    
    if not np.any(muscle_mask):
        features = {
            'Deep_Fat_Ratio': 0.0,
            'Fat_Entropy': 0.0,
            'Radial_FIP_Ring1': 0.0,
            'Radial_FIP_Ring2': 0.0,
            'Radial_FIP_Ring3': 0.0,
            'Fat_Centroid_Offset': 0.0,
            'Fat_Clustering_Index': 0.0,
            'Fascial_Fat_Ratio': 0.0
        }
        return features
    
    y_muscle, x_muscle = np.where(muscle_mask)
    muscle_centroid_y = np.mean(y_muscle)
    muscle_centroid_x = np.mean(x_muscle)
    
    muscle_points = np.column_stack([x_muscle, y_muscle])
    muscle_centroid = np.array([muscle_centroid_x, muscle_centroid_y])
    distances = np.linalg.norm(muscle_points - muscle_centroid, axis=1)
    sorted_distances = np.sort(distances)
    n_points = len(sorted_distances)
    
    features['Radial_FIP_Ring1'] = 0.0
    features['Radial_FIP_Ring2'] = 0.0
    features['Radial_FIP_Ring3'] = 0.0
    
    if n_points > 3:
        ring1_threshold = sorted_distances[int(n_points * 0.33)]
        ring2_threshold = sorted_distances[int(n_points * 0.66)]
        
        ring1_mask = np.zeros_like(muscle_mask, dtype=np.bool_)
        ring2_mask = np.zeros_like(muscle_mask, dtype=np.bool_)
        ring3_mask = np.zeros_like(muscle_mask, dtype=np.bool_)
        
        for i, (x, y) in enumerate(zip(x_muscle, y_muscle)):
            dist = distances[i]
            if dist <= ring1_threshold:
                ring1_mask[y, x] = True
            elif dist <= ring2_threshold:
                ring2_mask[y, x] = True
            else:
                ring3_mask[y, x] = True
        
        for name, ring in zip(['Radial_FIP_Ring1', 'Radial_FIP_Ring2', 'Radial_FIP_Ring3'], 
                            [ring1_mask, ring2_mask, ring3_mask]):
            ring_pixels = np.sum(ring)
            if ring_pixels > 0:
                ring_fat = np.sum(fat_mask & ring)
                features[name] = ring_fat / ring_pixels
            else:
                features[name] = 0.0
    
    edge_distances = ndimage.distance_transform_edt(muscle_mask.astype(np.uint8))
    max_distance = np.max(edge_distances)
    
    features['Deep_Fat_Ratio'] = 0.0
    features['Fascial_Fat_Ratio'] = 0.0
    
    if max_distance > 0:
        deep_threshold = max_distance * 0.33
        fascial_threshold = max_distance * 0.66
        
        # 肌肉外部距离为零，向肌肉中心逐渐增大，因此所有分区都必须与
        # 肌肉掩膜相交，避免把背景像素计入分母。
        shallow_zone = muscle_mask & (edge_distances <= deep_threshold)
        middle_zone = muscle_mask & (edge_distances > deep_threshold) & (edge_distances < fascial_threshold)
        deep_zone = muscle_mask & (edge_distances >= fascial_threshold)
        
        deep_fat = np.sum(fat_mask & deep_zone)
        deep_total = np.sum(deep_zone)
        shallow_fat = np.sum(fat_mask & shallow_zone)
        shallow_total = np.sum(shallow_zone)
        
        if deep_total > 0 and shallow_total > 0:
            # 深部区域内的脂肪占比，理论范围为 [0, 1]。
            features['Deep_Fat_Ratio'] = deep_fat / deep_total
        
        fascial_fat = np.sum(fat_mask & shallow_zone)
        fascial_total = np.sum(shallow_zone)
        if fascial_total > 0:
            features['Fascial_Fat_Ratio'] = fascial_fat / fascial_total
    
    fat_binary = fat_mask.astype(np.float32)
    if np.any(muscle_mask):
        histogram, _ = np.histogram(fat_binary[muscle_mask], bins=2, range=[0, 1])
        probabilities = histogram / np.sum(histogram)
        probabilities = probabilities[probabilities > 0]
        features['Fat_Entropy'] = -np.sum(probabilities * np.log2(probabilities)) if len(probabilities) > 0 else 0.0
    else:
        features['Fat_Entropy'] = 0.0
    
    y_fat, x_fat = np.where(fat_mask)
    if len(y_fat) > 0:
        fat_centroid_y = np.mean(y_fat)
        fat_centroid_x = np.mean(x_fat)
        
        offset_pixels = np.sqrt((fat_centroid_x - muscle_centroid_x)**2 + 
                                (fat_centroid_y - muscle_centroid_y)**2)
        offset_mm = offset_pixels * np.mean(pixel_spacing)
        
        area_mm2 = np.sum(muscle_mask) * (dx * dy)
        equivalent_radius = np.sqrt(area_mm2 / np.pi) if area_mm2 > 0 else 1.0
        features['Fat_Centroid_Offset'] = offset_mm / equivalent_radius
    else:
        features['Fat_Centroid_Offset'] = 0.0
    
    fat_connected = morphology.label(fat_mask.astype(np.uint8))
    if np.max(fat_connected) > 0:
        region_props = measure.regionprops(fat_connected)
        largest_area = max([prop.area for prop in region_props]) if region_props else 0
        total_fat = np.sum(fat_mask)
        features['Fat_Clustering_Index'] = largest_area / total_fat if total_fat > 0 else 0.0
    else:
        features['Fat_Clustering_Index'] = 0.0
    
    return features


def calculate_texture_features(
    normalized_image: np.ndarray,
    muscle_mask: np.ndarray,
    pixel_spacing: tuple
) -> dict:
    """
    璁＄畻 2.4 绾圭悊鐗瑰緛锛堜娇鐢?PyRadiomics锛?    
    Args:
        normalized_image: 鏍囧噯鍖栧悗鐨勫浘鍍?(H, W)
        muscle_mask: 鑲岃倝浜屽€兼帺鐮?(H, W)
        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛 (dx, dy)
    
    Returns:
        绾圭悊鐗瑰緛瀛楀吀
    """
    features = {}
    
    if not np.any(muscle_mask):
        features['Texture_FirstOrder_Entropy'] = 0.0
        features['Texture_FirstOrder_Skewness'] = 0.0
        features['Texture_FirstOrder_Kurtosis'] = 0.0
        features['Texture_GLCM_Contrast'] = 0.0
        features['Texture_GLCM_Correlation'] = 0.0
        features['Texture_GLCM_Id'] = 0.0
        features['Texture_GLCM_Idm'] = 0.0
        features['Texture_GLRLM_ShortRunEmphasis'] = 0.0
        features['Texture_GLRLM_LongRunEmphasis'] = 0.0
        features['Texture_GLRLM_RunLengthNonUniformity'] = 0.0
        features['Texture_GLSZM_SmallAreaEmphasis'] = 0.0
        features['Texture_GLSZM_LargeAreaEmphasis'] = 0.0
        features['Texture_GLSZM_SizeZoneNonUniformity'] = 0.0
        features['Texture_GLDM_DependenceEntropy'] = 0.0
        features['Texture_GLDM_DependenceNonUniformity'] = 0.0
        features['Texture_GLDM_GrayLevelNonUniformity'] = 0.0
        return features
    
    if not RADIOMICS_AVAILABLE:
        features['Texture_FirstOrder_Entropy'] = 0.0
        features['Texture_FirstOrder_Skewness'] = 0.0
        features['Texture_FirstOrder_Kurtosis'] = 0.0
        features['Texture_GLCM_Contrast'] = 0.0
        features['Texture_GLCM_Correlation'] = 0.0
        features['Texture_GLCM_Id'] = 0.0
        features['Texture_GLCM_Idm'] = 0.0
        features['Texture_GLRLM_ShortRunEmphasis'] = 0.0
        features['Texture_GLRLM_LongRunEmphasis'] = 0.0
        features['Texture_GLRLM_RunLengthNonUniformity'] = 0.0
        features['Texture_GLSZM_SmallAreaEmphasis'] = 0.0
        features['Texture_GLSZM_LargeAreaEmphasis'] = 0.0
        features['Texture_GLSZM_SizeZoneNonUniformity'] = 0.0
        features['Texture_GLDM_DependenceEntropy'] = 0.0
        features['Texture_GLDM_DependenceNonUniformity'] = 0.0
        features['Texture_GLDM_GrayLevelNonUniformity'] = 0.0
        return features
    
    try:
        import SimpleITK as sitk
        sitk_image = sitk.GetImageFromArray(normalized_image.astype(np.float32))
        # NumPy 切片生成的是二维 SimpleITK 图像，因此 spacing 也必须是二维。
        # 原先传入三个 spacing 值会在部分 SimpleITK 版本中触发异常回退。
        spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
        sitk_image.SetSpacing(spacing)

        sitk_mask = sitk.GetImageFromArray(muscle_mask.astype(np.uint8))
        sitk_mask.SetSpacing(spacing)

        params = {
            # CSF 归一化后的信号通常约为 0～2；原 binWidth=25 会把绝大多数
            # 肌肉区域压缩为单一灰度级，使纹理矩阵退化。
            'binWidth': 0.025,
            'normalize': False,
            'force2D': True,
            'verbose': False,
            'shape': 'False',
            'shape2D': 'False'
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(** params)
        extractor.enableImageTypes(Original={})
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(
            firstorder=['Entropy', 'Skewness', 'Kurtosis'],
            glcm=['Contrast', 'Correlation', 'Id', 'Idm'],
            glrlm=['ShortRunEmphasis', 'LongRunEmphasis', 'RunLengthNonUniformity'],
            glszm=['SmallAreaEmphasis', 'LargeAreaEmphasis', 'SizeZoneNonUniformity'],
            gldm=['DependenceEntropy', 'DependenceNonUniformity', 'GrayLevelNonUniformity']
        )

        radiomics_features = extractor.execute(sitk_image, sitk_mask)

        feature_mapping = {
            'original_firstorder_Entropy': 'Texture_FirstOrder_Entropy',
            'original_firstorder_Skewness': 'Texture_FirstOrder_Skewness',
            'original_firstorder_Kurtosis': 'Texture_FirstOrder_Kurtosis',
            'original_glcm_Contrast': 'Texture_GLCM_Contrast',
            'original_glcm_Correlation': 'Texture_GLCM_Correlation',
            'original_glcm_Id': 'Texture_GLCM_Id',
            'original_glcm_Idm': 'Texture_GLCM_Idm',
            'original_glrlm_ShortRunEmphasis': 'Texture_GLRLM_ShortRunEmphasis',
            'original_glrlm_LongRunEmphasis': 'Texture_GLRLM_LongRunEmphasis',
            'original_glrlm_RunLengthNonUniformity': 'Texture_GLRLM_RunLengthNonUniformity',
            'original_glszm_SmallAreaEmphasis': 'Texture_GLSZM_SmallAreaEmphasis',
            'original_glszm_LargeAreaEmphasis': 'Texture_GLSZM_LargeAreaEmphasis',
            'original_glszm_SizeZoneNonUniformity': 'Texture_GLSZM_SizeZoneNonUniformity',
            'original_gldm_DependenceEntropy': 'Texture_GLDM_DependenceEntropy',
            'original_gldm_DependenceNonUniformity': 'Texture_GLDM_DependenceNonUniformity',
            'original_gldm_GrayLevelNonUniformity': 'Texture_GLDM_GrayLevelNonUniformity'
        }

        for radiomics_key, feature_key in feature_mapping.items():
            if radiomics_key in radiomics_features:
                features[feature_key] = float(radiomics_features[radiomics_key])
            else:
                features[feature_key] = 0.0

    except Exception as e:
        print(f'Warning: Error in texture feature calculation: {e}')
        features['Texture_FirstOrder_Entropy'] = 0.0
        features['Texture_FirstOrder_Skewness'] = 0.0
        features['Texture_FirstOrder_Kurtosis'] = 0.0
        features['Texture_GLCM_Contrast'] = 0.0
        features['Texture_GLCM_Correlation'] = 0.0
        features['Texture_GLCM_Id'] = 0.0
        features['Texture_GLCM_Idm'] = 0.0
        features['Texture_GLRLM_ShortRunEmphasis'] = 0.0
        features['Texture_GLRLM_LongRunEmphasis'] = 0.0
        features['Texture_GLRLM_RunLengthNonUniformity'] = 0.0
        features['Texture_GLSZM_SmallAreaEmphasis'] = 0.0
        features['Texture_GLSZM_LargeAreaEmphasis'] = 0.0
        features['Texture_GLSZM_SizeZoneNonUniformity'] = 0.0
        features['Texture_GLDM_DependenceEntropy'] = 0.0
        features['Texture_GLDM_DependenceNonUniformity'] = 0.0
        features['Texture_GLDM_GrayLevelNonUniformity'] = 0.0

    return features


def calculate_intensity_features(
    normalized_image: np.ndarray,
    muscle_mask: np.ndarray,
    fat_mask: np.ndarray,
    pixel_spacing: tuple
) -> dict:
    """
    璁＄畻 2.2 鐏板害/淇″彿鐗瑰緛锛?4椤癸級
    
    Args:
        normalized_image: 鏍囧噯鍖栧悗鐨勫浘鍍?        muscle_mask: 鑲岃倝浜屽€兼帺鐮?        fat_mask: 鑴傝偑浜屽€兼帺鐮侊紙鍩轰簬鍔ㄦ€侀槇鍊硷級
        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛 (dx, dy)
    
    Returns:
        鐏板害鐗瑰緛瀛楀吀
    """
    features = {}
    dx, dy = pixel_spacing
    pixel_area = dx * dy
    
    muscle_pixels = normalized_image[muscle_mask]
    
    if len(muscle_pixels) == 0:
        features.update({
            'Mean_Intensity_Muscle': 0.0,
            'Std_Intensity_Muscle': 0.0,
            'Median_Intensity_Muscle': 0.0,
            'IQR_Intensity_Muscle': 0.0,
            'Skewness_Intensity_Muscle': 0.0,
            'Kurtosis_Intensity_Muscle': 0.0,
            'Mean_Intensity_Fat': 0.0,
            'Fat_Area': 0.0,
            'FIP': 0.0,
            'Lean_Muscle_Area': 0.0,
            'Mean_Intensity_Lean_Muscle': 0.0,
            'Fat_to_Lean_Ratio': 0.0,
            'Func_CSA': 0.0
        })
        return features
    
    features['Mean_Intensity_Muscle'] = np.mean(muscle_pixels)
    features['Std_Intensity_Muscle'] = np.std(muscle_pixels)
    features['Median_Intensity_Muscle'] = np.median(muscle_pixels)
    q1, q3 = np.percentile(muscle_pixels, [25, 75])
    features['IQR_Intensity_Muscle'] = q3 - q1
    
    if len(muscle_pixels) >= 3:
        features['Skewness_Intensity_Muscle'] = stats.skew(muscle_pixels)
    else:
        features['Skewness_Intensity_Muscle'] = 0.0
    
    if len(muscle_pixels) >= 4:
        features['Kurtosis_Intensity_Muscle'] = stats.kurtosis(muscle_pixels)
    else:
        features['Kurtosis_Intensity_Muscle'] = 0.0
    
    fat_pixels = normalized_image[fat_mask]
    if len(fat_pixels) > 0:
        features['Mean_Intensity_Fat'] = np.mean(fat_pixels)
    else:
        features['Mean_Intensity_Fat'] = 0.0
    
    features['Fat_Area'] = np.sum(fat_mask) * pixel_area
    total_muscle_pixels = np.sum(muscle_mask)
    if total_muscle_pixels > 0:
        features['FIP'] = np.sum(fat_mask) / total_muscle_pixels
    else:
        features['FIP'] = 0.0
    
    lean_mask = muscle_mask & ~fat_mask
    lean_muscle_area = np.sum(lean_mask) * pixel_area
    features['Lean_Muscle_Area'] = lean_muscle_area
    features['Func_CSA'] = lean_muscle_area
    
    lean_pixels = normalized_image[lean_mask]
    if len(lean_pixels) > 0:
        features['Mean_Intensity_Lean_Muscle'] = np.mean(lean_pixels)
    else:
        features['Mean_Intensity_Lean_Muscle'] = 0.0
    
    mean_fat = features['Mean_Intensity_Fat']
    mean_lean = features['Mean_Intensity_Lean_Muscle']
    if mean_lean > 0:
        features['Fat_to_Lean_Ratio'] = mean_fat / mean_lean
    else:
        features['Fat_to_Lean_Ratio'] = 0.0
    
    return features


def calculate_all_muscle_features(
    normalized_image: np.ndarray,
    muscle_mask: np.ndarray,
    pixel_spacing: tuple,
    fat_threshold: float
) -> dict:
    """
    璁＄畻鍗曚釜鑲岃倝鐨勬墍鏈夌壒寰侊紙褰㈡€佸 + 鐏板害 + 绌洪棿鍒嗗竷 + 绾圭悊锛?    
    Args:
        normalized_image: 鏍囧噯鍖栧悗鐨勫浘鍍?        muscle_mask: 鑲岃倝浜屽€兼帺鐮?        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛
        fat_threshold: 鑴傝偑鍍忕礌鍔ㄦ€侀槇鍊?    
    Returns:
        瀹屾暣鐗瑰緛瀛楀吀
    """
    fat_mask = muscle_mask & (normalized_image >= fat_threshold)
    
    morph_features = calculate_morphological_features(muscle_mask, pixel_spacing)
    intensity_features = calculate_intensity_features(normalized_image, muscle_mask, fat_mask, pixel_spacing)
    spatial_features = calculate_spatial_features(muscle_mask, fat_mask, pixel_spacing)
    texture_features = calculate_texture_features(normalized_image, muscle_mask, pixel_spacing)
    
    all_features = {}
    all_features.update(morph_features)
    all_features.update(intensity_features)
    all_features.update(spatial_features)
    all_features.update(texture_features)
    
    return all_features


def _fat_statistics_for_slice(image_2d, mask_2d, pixel_spacing):
    """返回单层区域的面积、功能面积、FIP、阈值及阈值决策类型。"""
    pixel_area = float(pixel_spacing[0]) * float(pixel_spacing[1])
    area = float(np.sum(mask_2d)) * pixel_area
    muscle_pixels = image_2d[mask_2d]
    threshold, decision = get_fat_threshold_per_muscle(muscle_pixels)
    fat_mask = mask_2d & (image_2d >= threshold)
    fat_area = float(np.sum(fat_mask)) * pixel_area
    functional_area = area - fat_area
    fip = fat_area / area if area > 0 else 0.0
    return area, functional_area, fip, threshold, decision


def calculate_3d_features(
    volume_data: np.ndarray,
    label_data: np.ndarray,
    muscle_label: int,
    pixel_spacing: tuple,
    slice_thickness: float = 1.0
) -> dict:
    """
    璁＄畻 Level 2: 3D 浣撶Н涓庡叏灞€鐗瑰緛锛堣法灞傞潰鑱氬悎锛?
    鏍规嵁 鑵版绋冲畾鎬ч娴嬬殑鐗瑰緛宸ョ▼涓庡缓妯℃柟妗堬紙瀹屾暣鎵╁厖鐗堬級-v2.md 鐨勮姹傦紝
    灏嗘瘡灞傜殑2D鐗瑰緛鎸夎倢鑲夋暣鍚堜负3D鎸囨爣銆?
    Args:
        volume_data: 3D 鏍囧噯鍖栧浘鍍?(x, y, z)
        label_data: 3D 鍒嗗壊鏍囩 (x, y, z)
        muscle_label: 鑲岃倝鏍囩缂栧彿
        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛 (dx, dy)锛屽崟浣峬m
        slice_thickness: 灞傚帤锛屽崟浣峬m

    Returns:
        3D鐗瑰緛瀛楀吀
    """
    from sklearn.mixture import GaussianMixture
    from scipy.signal import find_peaks
    from skimage.filters import threshold_otsu

    features = {}

    muscle_mask_3d = label_data == muscle_label

    areas = []
    func_csa = []
    fips = []
    centroids_z = []
    perimeters = []

    for z in range(volume_data.shape[2]):
        mask_2d = muscle_mask_3d[:, :, z]
        if not np.any(mask_2d):
            continue

        image_2d = volume_data[:, :, z]

        dx, dy = pixel_spacing
        pixel_area = dx * dy

        area = np.sum(mask_2d) * pixel_area
        areas.append(area)

        _, lean_area, fip, _, _ = _fat_statistics_for_slice(
            image_2d, mask_2d, pixel_spacing
        )
        func_csa.append(lean_area)
        fips.append(fip)

        y_coords, x_coords = np.where(mask_2d)
        if len(x_coords) > 0:
            centroids_z.append(z)
            perimeter = measure.perimeter(mask_2d, neighborhood=8) * np.mean(pixel_spacing)
            perimeters.append(perimeter)

    if len(areas) == 0:
        features = {
            '3D_Volume': 0.0,
            '3D_Func_Volume': 0.0,
            '3D_FIP': 0.0,
            'SA_V': 0.0,
            '3D_Shape_Index': 0.0,
            'Mean_Area': 0.0,
            'Max_Area': 0.0,
            'Min_Area': 0.0,
            'Std_Area': 0.0,
            'Mean_Func_CSA': 0.0,
            'Max_Func_CSA': 0.0,
            'Min_Func_CSA': 0.0,
            'Mean_FIP': 0.0,
            'Max_FIP': 0.0,
            'Min_FIP': 0.0,
            'Std_FIP': 0.0,
            'CV_Area_Z': 0.0,
            'CV_FIP_Z': 0.0,
            'Peak_Area_Slice_Index': -1,
            'Peak_FIP_Slice_Index': -1,
        }
        return features

    areas = np.array(areas)
    func_csa = np.array(func_csa)
    fips = np.array(fips)
    centroids_z = np.array(centroids_z)
    perimeters = np.array(perimeters)

    total_volume = np.sum(areas) * slice_thickness
    total_func_volume = np.sum(func_csa) * slice_thickness
    total_fat_volume = total_volume - total_func_volume

    features['3D_Volume'] = total_volume / 1000.0
    features['3D_Func_Volume'] = total_func_volume / 1000.0
    features['3D_FIP'] = total_fat_volume / total_volume if total_volume > 0 else 0.0

    mean_area = np.mean(areas)
    features['Mean_Area'] = mean_area
    features['Max_Area'] = np.max(areas)
    features['Min_Area'] = np.min(areas)
    features['Std_Area'] = np.std(areas) if len(areas) > 1 else 0.0

    features['Mean_Func_CSA'] = np.mean(func_csa)
    features['Max_Func_CSA'] = np.max(func_csa)
    features['Min_Func_CSA'] = np.min(func_csa)

    features['Mean_FIP'] = np.mean(fips)
    features['Max_FIP'] = np.max(fips)
    features['Min_FIP'] = np.min(fips)
    features['Std_FIP'] = np.std(fips) if len(fips) > 1 else 0.0

    features['CV_Area_Z'] = np.std(areas) / mean_area if mean_area > 0 else 0.0
    features['CV_FIP_Z'] = np.std(fips) / np.mean(fips) if np.mean(fips) > 0 else 0.0

    features['Peak_Area_Slice_Index'] = int(centroids_z[np.argmax(areas)])
    features['Peak_FIP_Slice_Index'] = int(centroids_z[np.argmax(fips)])

    # 按暴露体素面及真实体素尺寸计算表面积，单位为 mm²。
    voxel_mask = muscle_mask_3d.astype(np.uint8)
    dx, dy = map(float, pixel_spacing)
    dz = float(slice_thickness)
    padded = np.pad(voxel_mask, 1)
    surface_area = 0.0
    surface_area += np.sum(np.diff(padded, axis=0) != 0) * dy * dz
    surface_area += np.sum(np.diff(padded, axis=1) != 0) * dx * dz
    surface_area += np.sum(np.diff(padded, axis=2) != 0) * dx * dy
    features['SA_V'] = surface_area / total_volume if total_volume > 0 else 0.0
    features['3D_Shape_Index'] = (
        (surface_area ** 3) / (36.0 * np.pi * total_volume ** 2)
        if total_volume > 0 else 0.0
    )

    return features


def calculate_cross_layer_gradient_features(
    volume_data: np.ndarray,
    label_data: np.ndarray,
    muscle_label: int,
    pixel_spacing: tuple,
    slice_thickness: float = 1.0
) -> dict:
    """
    璁＄畻 Level 3.1: 璺ㄥ眰姊害鐗瑰緛锛堢旱鍚戝彉鍖栫巼锛?
    Args:
        volume_data: 3D 鏍囧噯鍖栧浘鍍?(x, y, z)
        label_data: 3D 鍒嗗壊鏍囩 (x, y, z)
        muscle_label: 鑲岃倝鏍囩缂栧彿
        pixel_spacing: 鍍忕礌鐗╃悊闂磋窛 (dx, dy)
        slice_thickness: 灞傚帤

    Returns:
        璺ㄥ眰姊害鐗瑰緛瀛楀吀
    """
    from sklearn.mixture import GaussianMixture
    from scipy.signal import find_peaks
    from skimage.filters import threshold_otsu
    from scipy.stats import linregress

    features = {}
    muscle_mask_3d = label_data == muscle_label

    areas = []
    func_csa = []
    fips = []
    centroids_x = []
    centroids_y = []
    aspect_ratios = []
    slice_indices = []

    for z in range(volume_data.shape[2]):
        mask_2d = muscle_mask_3d[:, :, z]
        if not np.any(mask_2d):
            continue

        image_2d = volume_data[:, :, z]
        dx, dy = pixel_spacing
        pixel_area = dx * dy

        area = np.sum(mask_2d) * pixel_area
        areas.append(area)

        y_coords, x_coords = np.where(mask_2d)
        if len(x_coords) > 0:
            centroid_x = np.mean(x_coords) * dx
            centroid_y = np.mean(y_coords) * dy
            centroids_x.append(centroid_x)
            centroids_y.append(centroid_y)

            rows = (np.max(y_coords) - np.min(y_coords) + 1) * dy
            cols = (np.max(x_coords) - np.min(x_coords) + 1) * dx
            aspect_ratio = max(rows, cols) / min(rows, cols) if min(rows, cols) > 0 else 1.0
            aspect_ratios.append(aspect_ratio)

        _, lean_area, fip, _, _ = _fat_statistics_for_slice(
            image_2d, mask_2d, pixel_spacing
        )
        func_csa.append(lean_area)
        fips.append(fip)

        slice_indices.append(z)

    if len(slice_indices) < 2:
        features = {
            'FIP_Slope': 0.0,
            'Area_Z_Gradient': 0.0,
            'Func_Area_Z_Gradient': 0.0,
            'Centroid_Z_Drift': 0.0,
            'Shape_Z_Deformation': 0.0,
        }
        return features

    slice_indices = np.array(slice_indices)
    z_positions_mm = slice_indices.astype(float) * float(slice_thickness)
    areas = np.array(areas)
    func_csa = np.array(func_csa)
    fips = np.array(fips)

    slope, _, _, _, _ = linregress(z_positions_mm, fips)
    features['FIP_Slope'] = slope

    slope, _, _, _, _ = linregress(z_positions_mm, areas)
    features['Area_Z_Gradient'] = slope

    slope, _, _, _, _ = linregress(z_positions_mm, func_csa)
    features['Func_Area_Z_Gradient'] = slope

    if len(centroids_x) >= 2:
        start_x, end_x = centroids_x[0], centroids_x[-1]
        start_y, end_y = centroids_y[0], centroids_y[-1]
        drift = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    else:
        drift = 0.0
    features['Centroid_Z_Drift'] = drift

    if len(aspect_ratios) >= 2:
        slope, _, _, _, _ = linregress(z_positions_mm[:len(aspect_ratios)], aspect_ratios)
        features['Shape_Z_Deformation'] = slope
    else:
        features['Shape_Z_Deformation'] = 0.0

    return features


def calculate_multi_muscle_features(
    volume_data: np.ndarray,
    label_data: np.ndarray,
    pixel_spacing: tuple,
    slice_thickness: float = 1.0
) -> dict:
    """
    璁＄畻 Level 3.2-3.5: 澶氳倢鑲夊叧绯汇€佸弻渚т笉瀵圭О銆佽剛鑲崗鍚岀瓑鐥呬汉绾х壒寰?
    Args:
        volume_data: 3D 鏍囧噯鍖栧浘鍍?        label_data: 3D 鍒嗗壊鏍囩
        pixel_spacing: 鍍忕礌闂磋窛
        slice_thickness: 灞傚帤

    Returns:
        鐥呬汉绾х患鍚堢壒寰佸瓧鍏?    """
    from sklearn.mixture import GaussianMixture
    from scipy.signal import find_peaks
    from skimage.filters import threshold_otsu
    from scipy.stats import linregress

    features = {}

    def get_muscle_stats(muscle_label):
        mask_3d = label_data == muscle_label
        total_volume = 0.0
        total_func_volume = 0.0
        mean_fip = 0.0
        mean_area = 0.0
        mean_intensity = 0.0
        fips_per_slice = []
        intensities_per_slice = []

        for z in range(volume_data.shape[2]):
            mask_2d = mask_3d[:, :, z]
            if not np.any(mask_2d):
                continue

            image_2d = volume_data[:, :, z]
            dx, dy = pixel_spacing
            pixel_area = dx * dy
            area = np.sum(mask_2d) * pixel_area
            mean_area += area

            muscle_pixels = image_2d[mask_2d]
            mean_intensity += np.mean(muscle_pixels) if len(muscle_pixels) > 0 else 0.0

            _, func_area, fip, _, _ = _fat_statistics_for_slice(
                image_2d, mask_2d, pixel_spacing
            )

            total_volume += area * slice_thickness
            total_func_volume += func_area * slice_thickness
            fips_per_slice.append(fip)
            intensities_per_slice.append(np.mean(muscle_pixels) if len(muscle_pixels) > 0 else 0.0)

        num_slices = np.sum(np.any(mask_3d, axis=(0, 1)))
        if num_slices > 0:
            mean_fip = np.mean(fips_per_slice) if len(fips_per_slice) > 0 else 0.0
            mean_area = mean_area / num_slices
            mean_intensity = mean_intensity / num_slices

        return {
            'volume': total_volume / 1000.0,
            'func_volume': total_func_volume / 1000.0,
            'mean_fip': mean_fip,
            'mean_area': mean_area,
            'mean_intensity': mean_intensity,
            'fips_per_slice': fips_per_slice,
            'intensities_per_slice': intensities_per_slice
        }

    # 标签含义必须与 feature_extraction_v7.MUSCLE_LABELS 完全一致。
    psoas_left = get_muscle_stats(1)
    psoas_right = get_muscle_stats(2)
    mf_left = get_muscle_stats(3)
    mf_right = get_muscle_stats(4)
    es_left = get_muscle_stats(5)
    es_right = get_muscle_stats(6)

    posterior_func_volume = mf_left['func_volume'] + mf_right['func_volume'] + es_left['func_volume'] + es_right['func_volume']
    psoas_func_volume = psoas_left['func_volume'] + psoas_right['func_volume']

    features['Psoas_Posterior_Ratio'] = psoas_func_volume / posterior_func_volume if posterior_func_volume > 0 else 0.0

    mf_mean_area = (mf_left['mean_area'] + mf_right['mean_area']) / 2.0
    es_mean_area = (es_left['mean_area'] + es_right['mean_area']) / 2.0
    psoas_mean_area = (psoas_left['mean_area'] + psoas_right['mean_area']) / 2.0

    features['ES_MF_Area_Ratio'] = es_mean_area / mf_mean_area if mf_mean_area > 0 else 0.0
    features['Psoas_ES_Area_Ratio'] = psoas_mean_area / es_mean_area if es_mean_area > 0 else 0.0
    features['MF_Psoas_Area_Ratio'] = mf_mean_area / psoas_mean_area if psoas_mean_area > 0 else 0.0

    features['Symmetry_Index_Area_MF'] = mf_left['mean_area'] / mf_right['mean_area'] if mf_right['mean_area'] > 0 else 0.0
    features['Symmetry_Index_Area_ES'] = es_left['mean_area'] / es_right['mean_area'] if es_right['mean_area'] > 0 else 0.0
    features['Symmetry_Index_Area_Psoas'] = psoas_left['mean_area'] / psoas_right['mean_area'] if psoas_right['mean_area'] > 0 else 0.0

    features['Symmetry_Index_FIP_MF'] = mf_left['mean_fip'] / mf_right['mean_fip'] if mf_right['mean_fip'] > 0 else 0.0
    features['Symmetry_Index_FIP_ES'] = es_left['mean_fip'] / es_right['mean_fip'] if es_right['mean_fip'] > 0 else 0.0
    features['Symmetry_Index_FIP_Psoas'] = psoas_left['mean_fip'] / psoas_right['mean_fip'] if psoas_right['mean_fip'] > 0 else 0.0

    features['Posterior_Func_Area_Total'] = posterior_func_volume * 1000.0

    mf_mean_fip = (mf_left['mean_fip'] + mf_right['mean_fip']) / 2.0
    es_mean_fip = (es_left['mean_fip'] + es_right['mean_fip']) / 2.0
    psoas_mean_fip = (psoas_left['mean_fip'] + psoas_right['mean_fip']) / 2.0

    features['Diff_FIP_MF_ES'] = mf_mean_fip - es_mean_fip
    features['Rat_FIP_MF_Psoas'] = mf_mean_fip / psoas_mean_fip if psoas_mean_fip > 0 else 0.0
    features['FIP_ES_MF_Product'] = es_mean_fip * mf_mean_fip

    es_mean_intensity = (es_left['mean_intensity'] + es_right['mean_intensity']) / 2.0
    features['Mean_Intensity_ES_MF_Ratio'] = es_mean_intensity / mf_left['mean_intensity'] if mf_left['mean_intensity'] > 0 else 0.0

    common_slices = min(len(mf_left['fips_per_slice']), len(es_left['fips_per_slice']))
    if common_slices >= 2:
        mf_fips = mf_left['fips_per_slice'][:common_slices]
        es_fips = es_left['fips_per_slice'][:common_slices]
        corr, _ = stats.pearsonr(mf_fips, es_fips)
        features['Intermuscle_FIP_Correlation'] = corr
    else:
        features['Intermuscle_FIP_Correlation'] = 0.0

    return features
