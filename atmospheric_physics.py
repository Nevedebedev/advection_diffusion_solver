"""
Atmospheric physics calculations for stability and diffusion
Used in BOTH training data generation AND deployment
"""
import numpy as np

def stability_class_to_D(stability_class: str, 
                         wind_speed: float,
                         z0: float = 0.03,
                         z: float = 2.0) -> float:
    """
    Calculate diffusion coefficient from Pasquill stability class
    
    Args:
        stability_class: 'A', 'B', 'C', 'D', 'E', or 'F'
        wind_speed: Wind speed (m/s)
        z0: Surface roughness (m)
        z: Release height (m)
    
    Returns:
        D: Diffusion coefficient (m²/s)
    """
    kappa = 0.4  # von Karman constant
    u_star = (kappa * wind_speed) / np.log(z / z0)
    D_base = kappa * u_star * z
    
    # Stability factors (from literature)
    stability_factors = {
        'A': 10.0,   # Very unstable (strong mixing)
        'B': 5.0,    # Unstable
        'C': 2.5,    # Slightly unstable
        'D': 1.0,    # Neutral
        'E': 0.6,    # Slightly stable
        'F': 0.3     # Stable (weak mixing)
    }
    
    factor = stability_factors.get(stability_class, 1.0)
    D = D_base * factor
    
    return np.clip(D, 0.3, 5.0)


def get_pasquill_class(wind_speed: float,
                      solar_radiation: float,
                      cloud_cover: float,
                      is_daytime: bool) -> str:
    """
    Determine Pasquill stability class from meteorological conditions
    """
    if is_daytime:
        # Daytime: solar radiation dominates
        if solar_radiation > 700:  # Strong
            if wind_speed < 2: return 'A'
            elif wind_speed < 3: return 'A'
            elif wind_speed < 5: return 'B'
            elif wind_speed < 6: return 'C'
            else: return 'D'
        elif solar_radiation > 350:  # Moderate
            if wind_speed < 2: return 'B'
            elif wind_speed < 3: return 'B'
            elif wind_speed < 5: return 'C'
            elif wind_speed < 6: return 'C'
            else: return 'D'
        else:  # Weak
            if wind_speed < 5: return 'C'
            else: return 'D'
    else:
        # Nighttime: cloud cover + wind
        if cloud_cover > 50:
            return 'D'  # Overcast = neutral
        else:
            if wind_speed < 2.5: return 'F'
            elif wind_speed < 3.5: return 'E'
            else: return 'D'


def sample_meteorological_conditions(wind_speed: float, rng):
    """
    Sample realistic meteorological conditions
    Returns: (solar_radiation, cloud_cover, hour, is_daytime)
    """
    # Sample hour
    hour = rng.uniform(0, 24)
    is_daytime = 6 <= hour <= 18
    
    if is_daytime:
        # Sample stability class (weighted by frequency)
        stability_choices = ['D', 'C', 'B', 'A']
        stability_weights = [0.50, 0.30, 0.15, 0.05]
        stability = rng.choice(stability_choices, p=stability_weights)
        
        # Time factor for solar radiation
        time_factor = np.sin(np.pi * (hour - 6) / 12)
        
        # Solar radiation and cloud cover based on stability
        if stability == 'A':
            solar = rng.uniform(700, 1000) * time_factor
            cloud = rng.uniform(0, 20)
        elif stability == 'B':
            solar = rng.uniform(500, 700) * time_factor
            cloud = rng.uniform(0, 30)
        elif stability == 'C':
            solar = rng.uniform(300, 500) * time_factor
            cloud = rng.uniform(20, 50)
        else:  # D
            solar = rng.uniform(100, 400) * time_factor
            cloud = rng.uniform(40, 80)
    else:
        # Nighttime
        stability_choices = ['D', 'E', 'F']
        stability_weights = [0.50, 0.35, 0.15]
        stability = rng.choice(stability_choices, p=stability_weights)
        
        solar = 0.0
        if stability == 'F':
            cloud = rng.uniform(0, 30)
        elif stability == 'E':
            cloud = rng.uniform(20, 60)
        else:
            cloud = rng.uniform(50, 100)
    
    return solar, cloud, hour, is_daytime
