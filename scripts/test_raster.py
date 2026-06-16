#!/usr/bin/env python3
"""Test raster_svg function"""
import sys
sys.path.insert(0, '/home/michel/code/IconShop')

from scripts.gallery import raster_svg

# Test with a simple SVG
svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="50" r="40" fill="red"/>
</svg>'''
result = raster_svg(svg)
print('Result type:', type(result))
print('Starts with data:', result.startswith('data:image') if isinstance(result, str) else 'no')
print('Length:', len(result))
print('First 100 chars:', result[:100] if isinstance(result, str) else result)
