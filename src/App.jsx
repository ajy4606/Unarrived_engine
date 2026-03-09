import React, { useState, useEffect, useRef, useMemo, useCallback, Suspense } from 'react';
import { Canvas, useThree, extend, useFrame, createPortal } from '@react-three/fiber';
import { shaderMaterial, useFBO } from '@react-three/drei';
import * as THREE from 'three';

import blueNoiseUrl from './assets/blue-noise.png';

// ---------------------------------------------------------
// 1. 프래그먼트 셰이더 (Adobe RGB 관련 수학 로직 삭제)
// ---------------------------------------------------------
const fragmentShader = `
  precision highp float;
  precision highp sampler3D;

  in vec2 vUv;

  uniform sampler2D uTexture;
  uniform sampler2D uBlueNoise;
  uniform sampler3D uLUT3D;

  uniform float uTime;
  uniform float uEffectTime; 
  uniform int   uFrame;      

  uniform float uAspect;
  uniform float uDitherEnabled; 
  uniform float uNoiseDensity;
  uniform float uNoiseAmount;
  uniform float uContrast;
  uniform float uBrightness;
  uniform float uLevels;
  uniform float uSoftness;
  uniform vec3  uColor1;
  uniform vec3  uColor2;

  uniform vec2  uTexelSize;
  uniform bool  uEnableSharpen;
  uniform float uSharpenAmount;
  uniform bool  uEnableVignette;
  uniform float uVignetteStrength;

  uniform vec3  uLift;
  uniform float uGamma;
  uniform vec3  uGain;

  uniform bool  uCAEnable;
  uniform float uCAAmount;
  uniform int   uCAType;

  uniform bool  uEnableGrain;
  uniform float uGrainAmount;
  uniform float uGrainSize;

  uniform bool  uEnableAnamorphic;
  uniform float uFlareThreshold;
  uniform float uFlareAmount;
  uniform float uFlareWidth;
  uniform vec3  uFlareColor;
  uniform float uHalationThreshold;

  uniform bool  uEnableLUT;
  uniform float uLUTMix;

  uniform float uGlitch;
  uniform float uSuperposition;
  uniform float uFluidity;
  uniform float uMelting;
  uniform float uHalation;

  uniform bool  uIsExporting;
  uniform float uShadows;
  uniform float uHighlights;
  uniform float uACESMix;
  uniform bool  uLinearize;

  uniform bool  uColorMode;
  uniform float uSaturation;
  uniform float uColorTemp;
  uniform float uColorTint;

  uniform vec3  uShadowTint;
  uniform vec3  uHighlightTint;
  uniform float uSplitBalance;

  uniform float uCrystallize;
  uniform float uScanDisplace;
  uniform float uEdgeHarvest;
  uniform float uThreshCascade;
  uniform float uStipple;
  uniform float uInkSpread;
  uniform float uSortGlitch;
  uniform float uFreqPeel;
  uniform int   uFreqPeelMode;

  uniform float uHalftone;
  uniform float uHalftoneScale;
  uniform float uHalftoneAngle;
  uniform float uCrossHatch;
  uniform float uCrossHatchAngle;
  uniform float uCrossHatchScale;
  uniform float uScatter;

  uniform sampler2D uPatternTex;
  uniform bool      uPatternEnabled;
  uniform float     uPatternScale;
  uniform float     uPatternAngle;
  uniform float     uPatternIntensity;
  uniform int       uPatternChannel;
  uniform int       uPatternMode;

  vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
  }

  float getBayer(vec2 p) {
    p = floor(p);
    float b = mod(p.x, 2.0) * 8.0 + mod(p.y, 2.0) * 4.0;
    p = floor(p / 2.0);
    b += mod(p.x, 2.0) * 2.0 + mod(p.y, 2.0) * 1.0;
    return b / 16.0;
  }

  vec3 ACESFilm(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0);
  }

  vec3 Reinhard(vec3 x) { return x / (x + 1.0); }

  vec3 toneMap(vec3 lin) {
    return mix(Reinhard(lin), ACESFilm(lin), uACESMix);
  }

  vec3 applySaturation(vec3 col, float sat) {
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(lum), col, sat);
  }

  vec3 applyWhiteBalance(vec3 col, float temp, float tint) {
    col.r *= max(1.0 + temp  * 0.15, 0.0);
    col.b *= max(1.0 - temp  * 0.15, 0.0);
    col.g *= max(1.0 - tint  * 0.08, 0.0);
    col.r *= max(1.0 + tint  * 0.04, 0.0);
    col.b *= max(1.0 + tint  * 0.04, 0.0);
    return max(col, vec3(0.0));
  }

  vec3 applySplitToning(vec3 col, float luma) {
    float shadowMask    = pow(clamp(1.0 - luma, 0.0, 1.0), 2.0 + uSplitBalance);
    float highlightMask = pow(clamp(luma, 0.0, 1.0),       2.0 - uSplitBalance);
    col += uShadowTint    * shadowMask    * 0.3;
    col += uHighlightTint * highlightMask * 0.3;
    return col;
  }

  float ditherChannel(float val, float noise, float noiseAmt, float levels, float softness) {
    float noisy = clamp(val + (noise - 0.5) * noiseAmt, 0.0, 1.0);
    float lS    = noisy * max(levels - 1.0, 1.0);
    float d     = smoothstep(0.5 - softness, 0.5 + softness, fract(lS));
    return clamp((floor(lS) + d) / max(levels - 1.0, 1.0), 0.0, 1.0);
  }

  vec2 patternUv(vec2 uv, float scale, float angle) {
    vec2 centered = uv - 0.5;
    float c = cos(angle), s = sin(angle);
    centered = vec2(c * centered.x - s * centered.y,
                    s * centered.x + c * centered.y);
    return fract(centered * scale + 0.5);
  }

  float patternSample(vec2 uv) {
    vec4 p = texture(uPatternTex, uv);
    float v;
    if      (uPatternChannel == 1) v = p.r;
    else if (uPatternChannel == 2) v = p.g;
    else if (uPatternChannel == 3) v = p.b;
    else if (uPatternChannel == 4) v = 1.0 - dot(p.rgb, vec3(0.299,0.587,0.114));
    else                           v = dot(p.rgb, vec3(0.299,0.587,0.114));
    return v;
  }

  vec2 patternGradient(vec2 puv, float eps) {
    float px = patternSample(puv + vec2(eps, 0.0)) - patternSample(puv - vec2(eps, 0.0));
    float py = patternSample(puv + vec2(0.0, eps)) - patternSample(puv - vec2(0.0, eps));
    return vec2(px, py);
  }

  vec2 voronoiCellUv(vec2 uv, float scale) {
    vec2 aspectUv = vec2(uv.x * uAspect, uv.y);
    vec2 gv   = aspectUv * scale;
    vec2 cell = floor(gv);
    float minD   = 9999.0;
    vec2  bestUv = uv;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        vec2 nb     = cell + vec2(float(dx), float(dy));
        vec2 jitter = hash2(nb * 7.3131 + vec2(1.73, 2.91));
        vec2 center = nb + jitter * 0.85 + 0.07;
        float d = length(gv - center);
        if (d < minD) {
          minD = d;
          bestUv = vec2(center.x / (scale * uAspect), center.y / scale);
        }
      }
    }
    return bestUv;
  }

  float halftoneCell(vec2 uv, float luma, float scale, float angle) {
    float c = cos(angle), s = sin(angle);
    vec2 ruv  = vec2(c*uv.x - s*uv.y, s*uv.x + c*uv.y) * scale;
    float rowOff = floor(ruv.y) * 0.5;
    vec2  cell   = fract(ruv + vec2(rowOff, 0.0)) - 0.5;
    float dotR   = (1.0 - clamp(luma, 0.0, 1.0)) * 0.48;
    return step(length(cell), dotR);
  }

  float crossHatchCell(vec2 uv, float luma, float scale, float angle) {
    float c1 = cos(angle),         s1 = sin(angle);
    float c2 = cos(angle + 1.5708), s2 = sin(angle + 1.5708);
    float l1 = fract(dot(uv, vec2(c1, s1)) * scale);
    float l2 = fract(dot(uv, vec2(c2, s2)) * scale);
    float thick = (1.0 - clamp(luma, 0.0, 1.0)) * 0.5 + 0.04;
    float line1 = step(l1, thick);
    float line2 = step(l2, thick * 0.55);
    return clamp(line1 + line2, 0.0, 1.0);
  }

  float scatterDot(vec2 bnBase, vec2 nOff, float luma) {
    float bn1 = texture(uBlueNoise, bnBase              + nOff).r;
    float bn2 = texture(uBlueNoise, bnBase * 0.4        + nOff + vec2(0.31, 0.67)).r;
    float bn3 = texture(uBlueNoise, bnBase * 2.3        + nOff + vec2(0.72, 0.19)).r;
    float thresh = bn1 * 0.55 + bn2 * 0.30 + bn3 * 0.15;
    return step(thresh, luma);
  }

  void main() {
    vec3  preColor = texture(uTexture, vUv).rgb;
    float preLuma  = dot(preColor, vec3(0.2126, 0.7152, 0.0722));
    vec2  modUv    = vUv;

    if (uGlitch > 0.0) {
      float mosh = step(0.95, fract(sin(floor(modUv.y * 60.0) * 12.345 + uEffectTime * 0.5) * 456.7));
      modUv.x += mosh * uGlitch * 0.05 * sin(uEffectTime * 15.0);
    }

    float patternMaskValue = 1.0;
    if (uPatternEnabled && uPatternMode == 0) {
      vec2  puv  = patternUv(modUv, uPatternScale, uPatternAngle);
      vec2  grad = patternGradient(puv, 0.004);
      modUv += grad * uPatternIntensity * 0.06;
      modUv  = clamp(modUv, 0.001, 0.999);
    }
    if (uPatternEnabled && uPatternMode == 1) {
      vec2 puv   = patternUv(modUv, uPatternScale, uPatternAngle);
      patternMaskValue = mix(1.0, patternSample(puv), uPatternIntensity);
    }
    if (uMelting > 0.0) {
      float safeTexel  = max(uTexelSize.x, 0.0001);
      float targetGrid = mix(1000.0, 10.0, uMelting);
      float actualGrid = mix(1.0 / safeTexel, targetGrid, (1.0 - preLuma) * uMelting);
      modUv = floor(modUv * actualGrid) / actualGrid;
    }
    if (uFluidity > 0.0) {
      vec2 h = hash2(modUv * 3.0 + vec2(uEffectTime * 0.2, -uEffectTime * 0.2));
      modUv += (h - 0.5) * uFluidity * 0.03 * (1.0 - preLuma);
    }

    if (uCrystallize > 0.0) {
      float eff    = uCrystallize * patternMaskValue;
      float scale  = exp(mix(log(90.0), log(3.0), eff));
      vec2 cellUv  = voronoiCellUv(modUv, scale);
      modUv        = mix(modUv, cellUv, eff);
    }

    if (uScanDisplace > 0.0) {
      float eff    = uScanDisplace * patternMaskValue;
      float origL  = dot(texture(uTexture, modUv).rgb, vec3(0.2126, 0.7152, 0.0722));
      float row    = floor(modUv.y / max(uTexelSize.y, 0.00001));
      float rnd    = fract(sin(row * 127.1 + floor(uEffectTime * 3.0) * 43.7) * 43758.5);
      float shift  = (rnd - 0.5) * eff * 0.12 + (origL - 0.5) * eff * 0.04;
      modUv.x      = fract(modUv.x + shift);
    }

    if (uThreshCascade > 0.0) {
      float eff         = uThreshCascade * patternMaskValue;
      float origL       = dot(texture(uTexture, modUv).rgb, vec3(0.2126, 0.7152, 0.0722));
      float shadowBand  = 1.0 - smoothstep(0.0, 0.4, origL);
      float hlBand      = smoothstep(0.6, 1.0, origL);
      float off         = eff * 0.016;
      vec2  cascadeOff  = vec2((hlBand - shadowBand) * off, (shadowBand - hlBand) * off * 0.6);
      modUv = clamp(modUv + cascadeOff, 0.001, 0.999);
    }

    if (uSortGlitch > 0.0) {
      float eff    = uSortGlitch * patternMaskValue;
      float myL    = dot(texture(uTexture, modUv).rgb, vec3(0.299, 0.587, 0.114));
      int   window = int(eff * 14.0) + 2;
      float maxL   = myL;
      int   maxIdx = 0;
      for (int i = 1; i <= 16; i++) {
        if (i > window) break;
        vec2  sampleUv = modUv + vec2(float(i) * uTexelSize.x, 0.0);
        float nbL      = dot(texture(uTexture, sampleUv).rgb, vec3(0.299, 0.587, 0.114));
        if (nbL > maxL) { maxL = nbL; maxIdx = i; }
      }
      float brightGain  = max(maxL - myL, 0.0);
      float sortMask    = smoothstep(0.15, 0.5, myL) * (1.0 - smoothstep(0.65, 0.9, myL));
      float targetShift = float(maxIdx) * uTexelSize.x;
      modUv.x = mix(modUv.x, fract(modUv.x + targetShift), eff * sortMask * brightGain * 3.0);
    }

    vec2  centerDir = modUv - 0.5;
    centerDir.x    *= uAspect;
    float dist      = length(centerDir);

    vec3 rawRgb = vec3(0.0);
    if (uCAEnable) {
      float caScale = (uCAType == 0) ? dist : smoothstep(0.4, 1.0, dist);
      vec2  delta   = modUv - 0.5;
      vec2  dir     = length(delta) > 0.0001 ? normalize(delta) : vec2(0.0);
      float amt     = uCAAmount * caScale;
      rawRgb.r = texture(uTexture, modUv + dir * amt).r;
      rawRgb.g = texture(uTexture, modUv + dir * amt * 0.5).g;
      rawRgb.b = texture(uTexture, modUv - dir * amt).b;
    } else {
      rawRgb = texture(uTexture, modUv).rgb;
    }

    if (uEnableSharpen) {
      vec3 c = rawRgb * 5.0;
      c -= texture(uTexture, modUv + vec2(-uTexelSize.x, 0.0)).rgb;
      c -= texture(uTexture, modUv + vec2( uTexelSize.x, 0.0)).rgb;
      c -= texture(uTexture, modUv + vec2(0.0,  uTexelSize.y)).rgb;
      c -= texture(uTexture, modUv + vec2(0.0, -uTexelSize.y)).rgb;
      rawRgb = mix(rawRgb, clamp(c, 0.0, 1.0), uSharpenAmount);
    }

    if (uFreqPeel > 0.0) {
      vec2 ts2 = uTexelSize * 2.0;
      vec3 blurred = (
        texture(uTexture, modUv + ts2*vec2(-1,-1)).rgb +
        texture(uTexture, modUv + ts2*vec2( 0,-1)).rgb * 2.0 +
        texture(uTexture, modUv + ts2*vec2( 1,-1)).rgb +
        texture(uTexture, modUv + ts2*vec2(-1, 0)).rgb * 2.0 +
        rawRgb * 4.0 +
        texture(uTexture, modUv + ts2*vec2( 1, 0)).rgb * 2.0 +
        texture(uTexture, modUv + ts2*vec2(-1, 1)).rgb +
        texture(uTexture, modUv + ts2*vec2( 0, 1)).rgb * 2.0 +
        texture(uTexture, modUv + ts2*vec2( 1, 1)).rgb
      ) / 16.0;
      vec3 hiFreq = clamp(rawRgb - blurred + 0.5, 0.0, 1.0); 
      vec3 loFreq = blurred;                                  
      if (uFreqPeelMode == 0) rawRgb = mix(rawRgb, hiFreq, uFreqPeel);
      else if (uFreqPeelMode == 1) rawRgb = mix(rawRgb, loFreq, uFreqPeel);
      else rawRgb = mix(rawRgb, 1.0 - hiFreq, uFreqPeel); 
    }

    float edgeMag = 0.0;
    if (uEdgeHarvest > 0.0) {
      vec2 ts = uTexelSize * 1.5;
      float p00 = dot(texture(uTexture, modUv + ts*vec2(-1,-1)).rgb, vec3(0.299,0.587,0.114));
      float p10 = dot(texture(uTexture, modUv + ts*vec2( 0,-1)).rgb, vec3(0.299,0.587,0.114));
      float p20 = dot(texture(uTexture, modUv + ts*vec2( 1,-1)).rgb, vec3(0.299,0.587,0.114));
      float p01 = dot(texture(uTexture, modUv + ts*vec2(-1, 0)).rgb, vec3(0.299,0.587,0.114));
      float p21 = dot(texture(uTexture, modUv + ts*vec2( 1, 0)).rgb, vec3(0.299,0.587,0.114));
      float p02 = dot(texture(uTexture, modUv + ts*vec2(-1, 1)).rgb, vec3(0.299,0.587,0.114));
      float p12 = dot(texture(uTexture, modUv + ts*vec2( 0, 1)).rgb, vec3(0.299,0.587,0.114));
      float p22 = dot(texture(uTexture, modUv + ts*vec2( 1, 1)).rgb, vec3(0.299,0.587,0.114));
      float gx  = -p00 - 2.0*p01 - p02 + p20 + 2.0*p21 + p22;
      float gy  = -p00 - 2.0*p10 - p20 + p02 + 2.0*p12 + p22;
      edgeMag   = clamp(length(vec2(gx, gy)) * 2.8, 0.0, 1.0);
    }

    vec3 halation = vec3(0.0);
    if (uHalation > 0.0) {
      vec2 offsets[4] = vec2[](vec2(1,0), vec2(-1,0), vec2(0,1), vec2(0,-1));
      float hL = dot(rawRgb, vec3(0.2126, 0.7152, 0.0722));
      halation += smoothstep(uHalationThreshold, 1.0, hL) * vec3(1.0, 0.1, 0.0);
      for (int i = 0; i < 4; i++) {
        vec3  s = texture(uTexture, modUv + offsets[i] * uTexelSize * 5.0).rgb;
        float l = dot(s, vec3(0.2126, 0.7152, 0.0722));
        halation += smoothstep(uHalationThreshold, 1.0, l) * vec3(1.0, 0.1, 0.0);
      }
      halation /= 5.0;
    }

    vec3 flare = vec3(0.0);
    if (uEnableAnamorphic) {
      for (int i = -5; i <= 5; i++) {
        vec2  fOff    = vec2(float(i) * uFlareWidth, 0.0);
        vec3  fSample = texture(uTexture, modUv + fOff).rgb;
        float b       = dot(fSample, vec3(0.2126, 0.7152, 0.0722));
        flare += smoothstep(uFlareThreshold, 1.0, b) * uFlareColor * (1.0 - abs(float(i)) / 5.0);
      }
    }

    vec3 lin = uLinearize ? pow(max(rawRgb, vec3(0.0)), vec3(2.2)) : max(rawRgb, vec3(0.0));
    lin += flare * uFlareAmount;
    lin += halation * uHalation;
    lin = applyWhiteBalance(lin, uColorTemp, uColorTint);
    lin = clamp(lin + uLift, 0.0, 10.0); 
    lin = pow(max(lin, vec3(0.0)), vec3(max(uGamma, 0.001))); 
    lin *= uGain;

    if (uEnableLUT) {
      vec3 lutColor = texture(uLUT3D, clamp(lin, 0.0, 1.0)).rgb;
      lin = mix(lin, lutColor, uLUTMix);
    }

    lin = applySaturation(lin, uSaturation);
    float currentLuma = dot(lin, vec3(0.2126, 0.7152, 0.0722));
    lin += smoothstep(0.25, 0.0, currentLuma) * uShadows * 0.5;

    float maxCh = max(lin.r, max(lin.g, lin.b));
    if (maxCh > 0.8) {
      float comp = (maxCh - 0.8) / (maxCh + 0.2);
      lin *= (1.0 - comp * 0.5);
    }
    
    lin = pow(max(lin, vec3(0.0)), vec3(1.0 / max(uHighlights, 0.01)));
    lin = toneMap(lin);

    float toningLuma = dot(lin, vec3(0.2126, 0.7152, 0.0722));
    lin = applySplitToning(lin, toningLuma);
    lin = clamp(lin, 0.0, 1.5);

    if (uEnableVignette) {
      float maxD      = length(vec2(0.5 * uAspect, 0.5));
      float vigRadius = maxD * 0.8;
      float vigSoft   = maxD * 0.5 * uVignetteStrength;
      lin *= smoothstep(vigRadius + vigSoft, max(0.001, vigRadius - vigSoft), dist);
      lin  = clamp(lin, 0.0, 1.5);
    }

    float luma;
    vec3  contrastCol;
    if (uColorMode) {
      contrastCol = clamp((lin - 0.5) * uContrast + 0.5 + uBrightness, 0.0, 1.0);
      luma        = dot(contrastCol, vec3(0.299, 0.587, 0.114));
    } else {
      luma        = dot(lin, vec3(0.299, 0.587, 0.114));
      luma        = clamp((luma - 0.5) * uContrast + 0.5 + uBrightness, 0.0, 1.0);
      contrastCol = vec3(luma);
    }

    vec2  nOff       = hash2(vec2(float(uFrame) * 0.137, float(uFrame) * 0.619)); 
    vec2  safeTexSz  = max(uTexelSize, vec2(0.00001));
    vec2  pixelCoord = modUv / safeTexSz;
    float noiseScale = uNoiseDensity * 0.1;               
    vec2  bnUvBase   = pixelCoord * noiseScale / 256.0;   

    float texNoise   = texture(uBlueNoise, bnUvBase + nOff).r;
    float bayerNoise = getBayer(pixelCoord * noiseScale);
    float noiseBase  = mix(texNoise, bayerNoise, uSuperposition);
    float noiseR     = mix(texture(uBlueNoise, bnUvBase + nOff + vec2(0.1,  0.3)).r, bayerNoise, uSuperposition);
    float noiseG     = noiseBase;
    float noiseB     = mix(texture(uBlueNoise, bnUvBase + nOff + vec2(0.5, -0.2)).r, bayerNoise, uSuperposition);

    if (uPatternEnabled && uPatternMode == 2) {
      vec2  puv     = patternUv(modUv, uPatternScale, uPatternAngle);
      float pVal    = patternSample(puv);
      noiseBase = mix(noiseBase, pVal, uPatternIntensity);
      noiseR    = mix(noiseR,    patternSample(patternUv(modUv, uPatternScale, uPatternAngle + 0.1)),  uPatternIntensity);
      noiseG    = mix(noiseG,    pVal, uPatternIntensity);
      noiseB    = mix(noiseB,    patternSample(patternUv(modUv, uPatternScale, uPatternAngle - 0.1)),  uPatternIntensity);
    }

    float safeNoiseAmt = uNoiseAmount;   
    float edgeSoftness = max(uSoftness, 0.05);

    if (uInkSpread > 0.0) {
      float eff    = uInkSpread * patternMaskValue;  
      float sD     = eff * 0.006;
      float nb0    = dot(texture(uTexture, modUv + vec2( sD, 0.0)).rgb, vec3(0.299,0.587,0.114));
      float nb1    = dot(texture(uTexture, modUv + vec2(-sD, 0.0)).rgb, vec3(0.299,0.587,0.114));
      float nb2    = dot(texture(uTexture, modUv + vec2(0.0,  sD)).rgb, vec3(0.299,0.587,0.114));
      float nb3    = dot(texture(uTexture, modUv + vec2(0.0, -sD)).rgb, vec3(0.299,0.587,0.114));
      float darkNb = min(min(nb0, nb1), min(nb2, nb3));
      luma         = mix(luma, min(luma, darkNb + 0.08), eff * 0.65);
      if (uColorMode) {
        contrastCol.r = mix(contrastCol.r, min(contrastCol.r, texture(uTexture, modUv + vec2( sD, 0.0)).r + 0.08), eff * 0.5);
        contrastCol.g = mix(contrastCol.g, min(contrastCol.g, texture(uTexture, modUv + vec2(0.0, -sD)).g + 0.08), eff * 0.5);
        contrastCol.b = mix(contrastCol.b, min(contrastCol.b, texture(uTexture, modUv + vec2(-sD, 0.0)).b + 0.08), eff * 0.5);
      }
    }

    if (uEdgeHarvest > 0.0) {
      float eff        = uEdgeHarvest * patternMaskValue;  
      edgeSoftness     = mix(edgeSoftness, 0.005, edgeMag * eff);
      float edgeFade   = mix(1.0, edgeMag + 0.08, eff * 0.85);
      float midFill    = 0.5;
      luma             = mix(luma, luma * edgeFade + midFill * (1.0 - edgeFade), eff * 0.75);
      if (uColorMode) {
        contrastCol = mix(contrastCol,
                          contrastCol * edgeFade + vec3(midFill) * (1.0 - edgeFade),
                          eff * 0.75);
      }
    }

    if (uStipple > 0.0) {
      float eff    = uStipple * patternMaskValue;
      edgeSoftness = mix(edgeSoftness, 0.001, eff);
      safeNoiseAmt = mix(safeNoiseAmt, 1.0, eff);  
      noiseBase    = mix(noiseBase, texNoise, eff * 0.8);
      noiseR       = mix(noiseR, texture(uBlueNoise, bnUvBase + nOff + vec2(0.13, 0.31)).r, eff * 0.8);
      noiseB       = mix(noiseB, texture(uBlueNoise, bnUvBase + nOff + vec2(0.47, -0.23)).r, eff * 0.8);
    }

    vec3 final;

    if (uDitherEnabled > 0.5) {
      if (uColorMode) {
        final.r = ditherChannel(contrastCol.r, noiseR, safeNoiseAmt, uLevels, edgeSoftness);
        final.g = ditherChannel(contrastCol.g, noiseG, safeNoiseAmt, uLevels, edgeSoftness);
        final.b = ditherChannel(contrastCol.b, noiseB, safeNoiseAmt, uLevels, edgeSoftness);
      } else {
        float noisyLuma = luma + (noiseBase - 0.5) * safeNoiseAmt;
        float lS        = clamp(noisyLuma, 0.0, 1.0) * max(uLevels - 1.0, 1.0);
        if (uGlitch > 0.0) {
          float mathErr = fract(sin(dot(modUv, vec2(12.9898, 78.233)) * (uEffectTime + 1.0)) * 43758.5453);
          if (mathErr > (1.0 - uGlitch * 0.1)) lS = mod(lS * (1.0 + uGlitch * 5.0), uLevels);
        }
        float d = smoothstep(0.5 - edgeSoftness, 0.5 + edgeSoftness, fract(lS));
        final   = mix(uColor1, uColor2, clamp((floor(lS) + d) / max(uLevels - 1.0, 1.0), 0.0, 1.0));
      }
    } else {
      if (uColorMode) {
        final = contrastCol;
      } else {
        final = mix(uColor1, uColor2, luma);
      }
    }

    if (uEnableGrain) {
      float grainMask = pow(clamp(luma * (1.0 - luma), 0.0, 1.0), 0.7);
      vec2  gSeed = modUv * uGrainSize + uEffectTime;
      float g     = fract(sin(dot(gSeed, vec2(12.9898, 78.233))) * 43758.5453);
      if (uColorMode) {
        float gR = fract(sin(dot(modUv * uGrainSize + uEffectTime + vec2(1.3, 2.7), vec2(12.9898, 78.233))) * 43758.5453);
        float gB = fract(sin(dot(modUv * uGrainSize + uEffectTime + vec2(5.1, 0.9), vec2(12.9898, 78.233))) * 43758.5453);
        final.r += (gR - 0.5) * uGrainAmount * grainMask;
        final.g += (g  - 0.5) * uGrainAmount * grainMask;
        final.b += (gB - 0.5) * uGrainAmount * grainMask;
      } else {
        final += (g - 0.5) * uGrainAmount * grainMask;
      }
    }

    if (uScatter > 0.0) {
      float dotPresence = scatterDot(bnUvBase, nOff, luma);
      vec3 scatterResult;
      if (uColorMode) {
        float dotR = scatterDot(bnUvBase + vec2(0.11, 0.23), nOff, contrastCol.r);
        float dotG = scatterDot(bnUvBase + vec2(0.47, 0.61), nOff, contrastCol.g);
        float dotB = scatterDot(bnUvBase + vec2(0.83, 0.37), nOff, contrastCol.b);
        scatterResult = vec3(dotR, dotG, dotB);
      } else {
        scatterResult = mix(uColor1, uColor2, dotPresence);
      }
      final = mix(final, scatterResult, uScatter);
    }

    if (uHalftone > 0.0) {
      if (uColorMode) {
        float htR = halftoneCell(modUv * vec2(uAspect, 1.0), contrastCol.r, uHalftoneScale, uHalftoneAngle);
        float htG = halftoneCell(modUv * vec2(uAspect, 1.0), contrastCol.g, uHalftoneScale, uHalftoneAngle + 0.2618);
        float htB = halftoneCell(modUv * vec2(uAspect, 1.0), contrastCol.b, uHalftoneScale, uHalftoneAngle + 0.5236);
        final.r = mix(final.r, mix(0.0, final.r * 1.4, htR), uHalftone);
        final.g = mix(final.g, mix(0.0, final.g * 1.4, htG), uHalftone);
        final.b = mix(final.b, mix(0.0, final.b * 1.4, htB), uHalftone);
      } else {
        float htDot = halftoneCell(modUv * vec2(uAspect, 1.0), luma, uHalftoneScale, uHalftoneAngle);
        vec3  htResult = mix(uColor2, uColor1, htDot);
        final = mix(final, htResult, uHalftone);
      }
    }

    if (uCrossHatch > 0.0) {
      if (uColorMode) {
        float hR = crossHatchCell(modUv * vec2(uAspect,1.0), contrastCol.r, uCrossHatchScale, uCrossHatchAngle);
        float hG = crossHatchCell(modUv * vec2(uAspect,1.0), contrastCol.g, uCrossHatchScale, uCrossHatchAngle + 1.047);
        float hB = crossHatchCell(modUv * vec2(uAspect,1.0), contrastCol.b, uCrossHatchScale, uCrossHatchAngle + 2.094);
        final.r = mix(final.r, final.r * mix(0.15, 1.6, hR), uCrossHatch);
        final.g = mix(final.g, final.g * mix(0.15, 1.6, hG), uCrossHatch);
        final.b = mix(final.b, final.b * mix(0.15, 1.6, hB), uCrossHatch);
      } else {
        float hatch    = crossHatchCell(modUv * vec2(uAspect, 1.0), luma, uCrossHatchScale, uCrossHatchAngle);
        vec3  chResult = mix(uColor2, uColor1, hatch);
        final = mix(final, chResult, uCrossHatch);
      }
    }

    if (uPatternEnabled && uPatternMode == 3) {
      vec2  puv   = patternUv(modUv, uPatternScale, uPatternAngle);
      float pVal  = patternSample(puv);
      vec3  soft  = mix(
        2.0 * final * pVal,
        1.0 - 2.0 * (1.0 - final) * (1.0 - pVal),
        step(0.5, pVal)
      );
      final = mix(final, clamp(soft, 0.0, 1.0), uPatternIntensity);
    }

    bvec3 isN = isnan(final);
    if (any(isN)) {
      final = vec3(0.0);
    }

    // ✨ Adobe RGB 로직 삭제, sRGB로 단순화
    if (uIsExporting) {
      gl_FragColor = vec4(pow(clamp(final, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
    } else {
      gl_FragColor = vec4(clamp(final, 0.0, 1.0), 1.0);
    }
  }
`;

const vertexShader = `
  out vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const blendFragmentShader = `
  precision highp float;
  in vec2 vUv;
  uniform sampler2D uCurrentFrame;
  uniform sampler2D uPreviousFrame;
  uniform float     uAccumulation;
  uniform bool      uIsFinalOutput;
  void main() {
    vec4 current  = texture(uCurrentFrame, vUv);
    vec4 previous = texture(uPreviousFrame, vUv);
    
    if (isnan(previous.r) || isnan(previous.g) || isnan(previous.b)) previous = current;
    if (isnan(current.r) || isnan(current.g) || isnan(current.b)) current = vec4(0.0);

    vec4 blended  = mix(current, previous, uAccumulation);
    if (uIsFinalOutput) {
      gl_FragColor = vec4(pow(clamp(blended.rgb, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
    } else {
      gl_FragColor = blended;
    }
  }
`;

// ---------------------------------------------------------
// 2. 셰이더 머티리얼 (Adobe RGB 관련 uniform 삭제)
// ---------------------------------------------------------
const DitherMaterial = shaderMaterial(
  {
    uTexture: null, uBlueNoise: null, uLUT3D: null,
    uTime: 0, uEffectTime: 0, uFrame: 0, uAspect: 1.0,
    uDitherEnabled: 1.0,
    uNoiseDensity: 30.0, uNoiseAmount: 0.5,
    uContrast: 1.0, uBrightness: 0.0,
    uLevels: 4.0, uSoftness: 0.1,
    uColor1: new THREE.Color('#111111'), uColor2: new THREE.Color('#f2f2f0'),
    uTexelSize: new THREE.Vector2(0, 0),
    uEnableSharpen: false, uSharpenAmount: 0.5,
    uEnableVignette: false, uVignetteStrength: 0.5,
    uLift: new THREE.Vector3(0, 0, 0), uGamma: 1.0, uGain: new THREE.Vector3(1, 1, 1),
    uCAEnable: false, uCAAmount: 0.015, uCAType: 0,
    uEnableGrain: true, uGrainAmount: 0.03, uGrainSize: 2.0,
    uEnableAnamorphic: false, uFlareThreshold: 0.9, uFlareAmount: 0.5,
    uFlareWidth: 0.01, uFlareColor: new THREE.Color('#4488ff'),
    uHalationThreshold: 0.65,
    uEnableLUT: false, uLUTMix: 1.0,
    uGlitch: 0.0, uSuperposition: 0.0, uFluidity: 0.0, uMelting: 0.0, uHalation: 0.0,
    uIsExporting: false,
    uShadows: 0.2, uHighlights: 1.0, uACESMix: 0.4, uLinearize: false,
    uColorMode: false, uSaturation: 1.0, uColorTemp: 0.0, uColorTint: 0.0,
    uShadowTint: new THREE.Vector3(0, 0, 0),
    uHighlightTint: new THREE.Vector3(0, 0, 0),
    uSplitBalance: 0.0,
    uCrystallize:   0.0,
    uScanDisplace:  0.0,
    uEdgeHarvest:   0.0,
    uThreshCascade: 0.0,
    uStipple:       0.0,
    uInkSpread:     0.0,
    uSortGlitch:    0.0,
    uFreqPeel:      0.0,
    uFreqPeelMode:  0,
    uHalftone:        0.0,
    uHalftoneScale:   40.0,
    uHalftoneAngle:   0.0,
    uCrossHatch:      0.0,
    uCrossHatchAngle: 0.0,
    uCrossHatchScale: 40.0,
    uScatter:         0.0,
    uPatternTex:       null,
    uPatternEnabled:   false,
    uPatternScale:     4.0,
    uPatternAngle:     0.0,
    uPatternIntensity: 0.7,
    uPatternChannel:   0,
    uPatternMode:      0,
  },
  vertexShader, fragmentShader
);
DitherMaterial.glslVersion = THREE.GLSL3;
extend({ DitherMaterial });

const BlendMaterial = shaderMaterial(
  { uCurrentFrame: null, uPreviousFrame: null, uAccumulation: 0.8, uIsFinalOutput: false },
  vertexShader, blendFragmentShader
);
BlendMaterial.glslVersion = THREE.GLSL3;
extend({ BlendMaterial });

// ---------------------------------------------------------
// 3. 엔진 코어
// ---------------------------------------------------------
const EngineCore = ({ texture, bNoise, lut3D, patternTex, params, aspect, saveRef, setIsExporting, setExportResult }) => {
  const { gl, size, camera } = useThree();

  const FOV_HEIGHT = 8.0;
  const canvasAspect = size.width / size.height;
  const planeW = aspect > canvasAspect ? FOV_HEIGHT * canvasAspect : FOV_HEIGHT * aspect;
  const planeH = aspect > canvasAspect ? FOV_HEIGHT * canvasAspect / aspect : FOV_HEIGHT;

  const renderScene = useMemo(() => new THREE.Scene(), []);
  const blendScene  = useMemo(() => new THREE.Scene(), []);
  const orthoCamera = useMemo(() => new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1), []);
  const ditherMatRef = useRef();
  const blendMatRef  = useRef();

  const fboSettings = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, type: THREE.HalfFloatType };
  const targetA    = useFBO(size.width, size.height, fboSettings);
  const targetB    = useFBO(size.width, size.height, fboSettings);
  const currentFbo = useFBO(size.width, size.height, fboSettings);
  const isTargetA    = useRef(true);
  
  const resetTaaFlag = useRef(true); 
  
  const frameRef = useRef(0);
  const effectTimeRef = useRef(0);
  const settleCountRef = useRef(0);

  useEffect(() => {
    resetTaaFlag.current = true;
    settleCountRef.current = 0;
  }, [params, texture, size]); 

  useEffect(() => {
    targetA.setSize(size.width, size.height);
    targetB.setSize(size.width, size.height);
    currentFbo.setSize(size.width, size.height);
  }, [size, targetA, targetB, currentFbo]);

  useEffect(() => { return () => { targetA.dispose(); targetB.dispose(); currentFbo.dispose(); }; }, [targetA, targetB, currentFbo]);

  const uTexelSize = useMemo(() => new THREE.Vector2(1 / (texture.image?.width ?? 1), 1 / (texture.image?.height ?? 1)), [texture]);

  useFrame((state, delta) => {
    if (!ditherMatRef.current || !blendMatRef.current) return;
    if (ditherMatRef.current.uniforms.uIsExporting.value) return;
    if (params.isExportingFlag) return;

    const justChanged = resetTaaFlag.current;
    resetTaaFlag.current = false;

    const isDirty = settleCountRef.current < 45; 

    if (params.animate) {
      effectTimeRef.current += delta;
      frameRef.current += 1;
      settleCountRef.current = 0; 
    } else {
      if (params.temporalEnabled && isDirty) {
        frameRef.current += 1; 
        settleCountRef.current += 1;
      }
    }

    const dm = ditherMatRef.current.uniforms;
    dm.uTime.value       = state.clock.elapsedTime;
    dm.uEffectTime.value = effectTimeRef.current;
    dm.uFrame.value      = frameRef.current;
    dm.uDitherEnabled.value = params.ditherEnabled ? 1.0 : 0.0;

    dm.uGlitch.value        = params.glitch;
    dm.uSuperposition.value = params.superposition;
    dm.uFluidity.value      = params.fluidity;
    dm.uMelting.value       = params.melting;
    dm.uHalation.value      = params.halation;
    dm.uShadows.value       = params.shadows;
    dm.uHighlights.value    = params.highlights;
    dm.uACESMix.value       = params.acesMix;
    dm.uLinearize.value     = params.linearize;
    dm.uColorMode.value     = params.colorMode;
    dm.uSaturation.value    = params.saturation;
    dm.uColorTemp.value     = params.colorTemp;
    dm.uColorTint.value     = params.colorTint;
    dm.uSplitBalance.value  = params.splitBalance;
    dm.uShadowTint.value.set(params.shadowTint.r, params.shadowTint.g, params.shadowTint.b);
    dm.uHighlightTint.value.set(params.highlightTint.r, params.highlightTint.g, params.highlightTint.b);
    dm.uContrast.value      = params.contrast;
    dm.uBrightness.value    = params.brightness;
    dm.uGamma.value         = params.gamma;
    dm.uLift.value.set(params.lift.r, params.lift.g, params.lift.b);
    dm.uGain.value.set(params.gain.r, params.gain.g, params.gain.b);
    dm.uLevels.value        = params.levels;
    dm.uSoftness.value      = params.softness;
    dm.uNoiseDensity.value  = params.density;
    dm.uNoiseAmount.value   = params.noiseAmount;
    dm.uColor1.value.set(params.color1);
    dm.uColor2.value.set(params.color2);
    dm.uEnableSharpen.value    = params.sharpen;
    dm.uSharpenAmount.value    = params.sharpenAmount;
    dm.uEnableVignette.value   = params.vignette;
    dm.uVignetteStrength.value = params.vignetteStrength;
    dm.uCAEnable.value         = params.caEnable;
    dm.uCAAmount.value         = params.caAmount;
    dm.uCAType.value           = params.caType;
    dm.uEnableGrain.value      = true;
    dm.uGrainAmount.value      = params.grain;
    dm.uGrainSize.value        = params.grainSize;
    dm.uEnableAnamorphic.value = params.flare;
    dm.uFlareThreshold.value   = params.flareThresh;
    dm.uFlareAmount.value      = params.flareAmount;
    dm.uFlareWidth.value       = params.flareWidth;
    dm.uFlareColor.value.set(params.flareColor);
    dm.uHalationThreshold.value = params.halationThresh;
    dm.uEnableLUT.value        = params.lutEnabled;
    dm.uLUTMix.value           = params.lutMix;
    dm.uCrystallize.value   = params.crystallize;
    dm.uScanDisplace.value  = params.scanDisplace;
    dm.uEdgeHarvest.value   = params.edgeHarvest;
    dm.uThreshCascade.value = params.threshCascade;
    dm.uStipple.value       = params.stipple;
    dm.uInkSpread.value     = params.inkSpread;
    dm.uSortGlitch.value    = params.sortGlitch;
    dm.uFreqPeel.value      = params.freqPeel;
    dm.uFreqPeelMode.value  = params.freqPeelMode;
    dm.uHalftone.value        = params.halftone;
    dm.uHalftoneScale.value   = params.halftoneScale;
    dm.uHalftoneAngle.value   = params.halftoneAngle;
    dm.uCrossHatch.value      = params.crossHatch;
    dm.uCrossHatchAngle.value = params.crossHatchAngle;
    dm.uCrossHatchScale.value = params.crossHatchScale;
    dm.uScatter.value         = params.scatter;
    dm.uPatternEnabled.value   = params.patternEnabled;
    dm.uPatternScale.value     = params.patternScale;
    dm.uPatternAngle.value     = params.patternAngle;
    dm.uPatternIntensity.value = params.patternIntensity;
    dm.uPatternChannel.value   = params.patternChannel;
    dm.uPatternMode.value      = params.patternMode;

    if (!params.temporalEnabled) {
      gl.setRenderTarget(currentFbo);
      gl.render(renderScene, camera);

      const bm = blendMatRef.current.uniforms;
      bm.uCurrentFrame.value  = currentFbo.texture;
      bm.uPreviousFrame.value = currentFbo.texture; 
      bm.uAccumulation.value  = 0.0;
      bm.uIsFinalOutput.value = true;

      gl.setRenderTarget(null);
      gl.render(blendScene, orthoCamera);
      return;
    }

    const writeBuffer = isTargetA.current ? targetA : targetB;
    const readBuffer  = isTargetA.current ? targetB : targetA;

    gl.setRenderTarget(currentFbo);
    gl.render(renderScene, camera);

    const bm = blendMatRef.current.uniforms;
    bm.uCurrentFrame.value  = currentFbo.texture;
    bm.uPreviousFrame.value = readBuffer.texture;
    bm.uAccumulation.value  = justChanged ? 0.0 : params.accumulation;
    bm.uIsFinalOutput.value = false;

    gl.setRenderTarget(writeBuffer);
    gl.render(blendScene, orthoCamera);

    bm.uCurrentFrame.value  = writeBuffer.texture;
    bm.uAccumulation.value  = 0.0;
    bm.uIsFinalOutput.value = true;

    gl.setRenderTarget(null);
    gl.render(blendScene, orthoCamera);
    isTargetA.current = !isTargetA.current;
  }, 1);

  return (
    <>
      <Exporter
        saveRef={saveRef} params={params}
        setIsExporting={setIsExporting} setExportResult={setExportResult}
        texture={texture} renderScene={renderScene}
      />
      {createPortal(
        <mesh>
          <planeGeometry args={[planeW, planeH]} />
          <ditherMaterial
            ref={ditherMatRef}
            uTexture={texture}
            uBlueNoise={bNoise}
            uLUT3D={lut3D}
            uPatternTex={patternTex}
            uAspect={aspect}
            uTexelSize={uTexelSize}
          />
        </mesh>,
        renderScene
      )}
      {createPortal(
        <mesh>
          <planeGeometry args={[2, 2]} />
          <blendMaterial ref={blendMatRef} />
        </mesh>,
        blendScene
      )}
    </>
  );
};

// ---------------------------------------------------------
// 4. Exporter
// ---------------------------------------------------------
const Exporter = ({ saveRef, params, setIsExporting, setExportResult, texture, renderScene }) => {
  const { gl, camera } = useThree();
  const paramsRef = useRef(params);
  useEffect(() => { paramsRef.current = params; }, [params]);

  useEffect(() => {
    saveRef.current = async () => {
      if (!texture) return;
      const p      = paramsRef.current;
      const aspect = texture.image?.width / texture.image?.height || 1;

      let outW, outH;
      if (aspect >= 1) { outW = p.resolution; outH = Math.round(p.resolution / aspect); }
      else             { outH = p.resolution; outW = Math.round(p.resolution * aspect); }

      setIsExporting(true);
      const fixedTime = performance.now() / 1000;

      renderScene.traverse(o => {
        const u = o.material?.uniforms;
        if (!u) return;
        if (u.uTime)           u.uTime.value           = fixedTime;
        if (u.uEffectTime)     u.uEffectTime.value     = fixedTime;
        if (u.uFrame)          u.uFrame.value          = 1;
        if (u.uIsExporting)    u.uIsExporting.value    = true;
      });

      const tileSize    = 2048;
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width  = outW; finalCanvas.height = outH;
      const finalCtx    = finalCanvas.getContext('2d');
      const tileCanvas  = document.createElement('canvas');
      const tileCtx     = tileCanvas.getContext('2d', { willReadFrequently: true });
      const pixelBuffer = new Uint8Array(tileSize * tileSize * 4);

      const exCam = camera.clone();
      if (exCam.isPerspectiveCamera) { exCam.aspect = aspect; exCam.updateProjectionMatrix(); }
      const rt = new THREE.WebGLRenderTarget(tileSize, tileSize);

      try {
        for (let y = 0; y < outH; y += tileSize) {
          for (let x = 0; x < outW; x += tileSize) {
            const cw = Math.min(tileSize, outW - x);
            const ch = Math.min(tileSize, outH - y);
            rt.setSize(cw, ch);
            exCam.setViewOffset(outW, outH, x, y, cw, ch);
            gl.setRenderTarget(rt);
            gl.render(renderScene, exCam);
            gl.readRenderTargetPixels(rt, 0, 0, cw, ch, pixelBuffer);

            const clamped = new Uint8ClampedArray(pixelBuffer.buffer, 0, cw * ch * 4);
            tileCanvas.width = cw; tileCanvas.height = ch;
            tileCtx.putImageData(new ImageData(clamped, cw, ch), 0, 0);

            finalCtx.save();
            finalCtx.translate(x, y + ch);
            finalCtx.scale(1, -1);
            finalCtx.drawImage(tileCanvas, 0, 0);
            finalCtx.restore();

            await new Promise(r => setTimeout(r, 10));
          }
        }

        const dataUrl = finalCanvas.toDataURL('image/png');
        setExportResult({
          dataUrl,
          w: outW,
          h: outH
        });

      } catch (err) {
        console.error('Export Error:', err);
        alert('이미지 추출 중 오류가 발생했습니다.\n' + err.message);

      } finally {
        exCam.clearViewOffset();
        gl.setRenderTarget(null);
        rt.dispose();
        renderScene.traverse(o => {
          const u = o.material?.uniforms;
          if (!u) return;
          if (u.uIsExporting) u.uIsExporting.value = false;
        });
        setIsExporting(false);
      }
    };
  }, [gl, camera, saveRef, setIsExporting, texture, renderScene, setExportResult]);

  return null;
};

// ---------------------------------------------------------
// 5. 메인 앱
// ---------------------------------------------------------
const PRESETS_KEY = 'unarrived_engine_v2';

const hexToRgbOffset = (hex) => ({
  r: parseInt(hex.slice(1, 3), 16) / 255 - 0.5,
  g: parseInt(hex.slice(3, 5), 16) / 255 - 0.5,
  b: parseInt(hex.slice(5, 7), 16) / 255 - 0.5,
});
const rgbOffsetToHex = ({ r, g, b }) => {
  const to8 = v => Math.max(0, Math.min(255, Math.round((v + 0.5) * 255)));
  return '#' + [r, g, b].map(v => to8(v).toString(16).padStart(2, '0')).join('');
};
const TINT_NEUTRAL = { r: 0, g: 0, b: 0 };

const INITIAL_PARAMS = {
  resolution: 4000,
  ditherEnabled: false,
  density: 30, noiseAmount: 0.0,
  contrast: 1.0, brightness: 0.0,
  shadows: 0.2, highlights: 1.0,
  acesMix: 0.4, linearize: false,
  colorMode: false,
  saturation: 1.0, colorTemp: 0.0, colorTint: 0.0,
  shadowTint: { ...TINT_NEUTRAL },
  highlightTint: { ...TINT_NEUTRAL },
  splitBalance: 0.0,
  levels: 4, softness: 0.1,
  color1: '#111111', color2: '#f2f2f0',
  sharpen: false, sharpenAmount: 0.5,
  vignette: false, vignetteStrength: 0.5,
  lift: { r: 0, g: 0, b: 0 }, gamma: 1.0, gain: { r: 1, g: 1, b: 1 },
  caEnable: false, caAmount: 0.015, caType: 0,
  grain: 0.03, grainSize: 2.0,
  flare: false, flareThresh: 0.9, flareAmount: 0.5,
  flareWidth: 0.01, flareColor: '#4488ff',
  halationThresh: 0.65,
  lutEnabled: false, lutMix: 1.0, lutDataUrl: null, lutName: null, lutSize: null,
  animate: false, 
  temporalEnabled: true,
  glitch: 0.0, superposition: 0.0, fluidity: 0.0, melting: 0.0, halation: 0.0,
  accumulation: 0.85,
  crystallize: 0.0, scanDisplace: 0.0, edgeHarvest: 0.0,
  threshCascade: 0.0, stipple: 0.0, inkSpread: 0.0,
  sortGlitch: 0.0, freqPeel: 0.0, freqPeelMode: 0,
  halftone: 0.0, halftoneScale: 40.0, halftoneAngle: 0.0,
  crossHatch: 0.0, crossHatchAngle: 0.0, crossHatchScale: 40.0,
  scatter: 0.0,
  patternEnabled: false, patternScale: 4.0, patternAngle: 0.0,
  patternIntensity: 0.7, patternChannel: 0, patternMode: 0,
  isExportingFlag: false,
};

export default function App() {
  const [params, setParams] = useState({ ...INITIAL_PARAMS });

  const [texture,      setTexture]      = useState(null);
  const [bNoise,       setBNoise]       = useState(null);
  const [lut3D,        setLut3D]        = useState(null);
  const [patternTex,   setPatternTex]   = useState(null);
  const [presets,      setPresets]      = useState([]);
  const [presetName,   setPresetName]   = useState('');
  const [webgl2,       setWebgl2]       = useState(true);
  const [exportResult, setExportResult] = useState(null);
  const [sidebarOpen,  setSidebarOpen]  = useState(false);
  const [isMobile,     setIsMobile]     = useState(() => window.innerWidth < 768);
  const [zoom,         setZoom]         = useState(1);
  const [pan,          setPan]          = useState({ x: 0, y: 0 });
  const [zoomVisible,  setZoomVisible]  = useState(false);
  const [isPanningState, setIsPanningState] = useState(false); 
  const isPanning = useRef(false);
  const panStart  = useRef({ x: 0, y: 0, px: 0, py: 0 });
  const zoomHideTimer = useRef(null);
  const canvasContainerRef = useRef(null);

  const clampPan = useCallback((px, py, z) => {
    if (!canvasContainerRef.current || z <= 1) return { x: 0, y: 0 };
    const { width, height } = canvasContainerRef.current.getBoundingClientRect();
    const maxX = (width  * (z - 1)) / 2;
    const maxY = (height * (z - 1)) / 2;
    return {
      x: Math.max(-maxX, Math.min(maxX, px)),
      y: Math.max(-maxY, Math.min(maxY, py)),
    };
  }, []);

  const saveRef = useRef();
  const lutRef  = useRef();
  const patternRef = useRef();

  const setIsExporting = useCallback((flag) => setParams(p => ({ ...p, isExportingFlag: flag })), []);
  const sp = useCallback((key) => (v) => setParams(p => ({ ...p, [key]: v })), []);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    const loader = new THREE.TextureLoader();
    loader.load(blueNoiseUrl,
      (tex) => {
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.minFilter = tex.magFilter = THREE.NearestFilter;
        tex.generateMipmaps = false;
        setBNoise(tex);
      },
      undefined,
      () => alert('🚨 blue-noise.png 로드 실패! src/assets/ 경로를 확인해주세요.')
    );
    loader.setCrossOrigin('anonymous');
    loader.load(
      'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1000&auto=format&fit=crop',
      setTexture
    );
    const saved = localStorage.getItem(PRESETS_KEY);
    if (saved) { try { setPresets(JSON.parse(saved)); } catch {} }
    const glCtx = document.createElement('canvas').getContext('webgl2');
    if (!glCtx) setWebgl2(false);
  }, []);

  const defaultLut3D = useMemo(() => {
    const t = new THREE.Data3DTexture(new Uint8Array([128, 128, 128, 255]), 1, 1, 1);
    t.format = THREE.RGBAFormat; t.minFilter = t.magFilter = THREE.LinearFilter;
    t.unpackAlignment = 1; t.needsUpdate = true;
    return t;
  }, []);

  const handleImageUpload = useCallback((e) => {
    const file = e.target.files[0]; if (!file) return;
    const url = URL.createObjectURL(file);
    new THREE.TextureLoader().load(url, (tex) => {
      setTexture(prev => { prev?.dispose(); return tex; });
      URL.revokeObjectURL(url);
    });
  }, []);

  const parseCubeLUT = useCallback((text) => {
    const lines = text.split('\n');
    let size = 0;
    const floats = [];

    for (const raw of lines) {
      const line = raw.trim();
      if (!line || line.startsWith('#') || line.startsWith('TITLE')) continue;
      if (line.startsWith('LUT_3D_SIZE')) {
        size = parseInt(line.split(/\s+/)[1], 10);
        continue;
      }
      if (line.startsWith('DOMAIN_MIN') || line.startsWith('DOMAIN_MAX') ||
          line.startsWith('LUT_1D_SIZE') || line.startsWith('LUT_2D_SIZE')) continue;
      const parts = line.split(/\s+/);
      if (parts.length >= 3) {
        const r = parseFloat(parts[0]), g = parseFloat(parts[1]), b = parseFloat(parts[2]);
        if (!isNaN(r) && !isNaN(g) && !isNaN(b)) floats.push(r, g, b);
      }
    }

    if (!size) throw new Error('LUT_3D_SIZE를 찾을 수 없습니다.');
    if (floats.length !== size * size * size * 3)
      throw new Error(`데이터 행 수 불일치: 예상 ${size ** 3}행, 실제 ${floats.length / 3 | 0}행`);

    const rgba = new Uint8Array(size * size * size * 4);
    for (let i = 0; i < size * size * size; i++) {
      rgba[i * 4]     = Math.round(Math.min(Math.max(floats[i * 3],     0), 1) * 255);
      rgba[i * 4 + 1] = Math.round(Math.min(Math.max(floats[i * 3 + 1], 0), 1) * 255);
      rgba[i * 4 + 2] = Math.round(Math.min(Math.max(floats[i * 3 + 2], 0), 1) * 255);
      rgba[i * 4 + 3] = 255;
    }
    const tex3d = new THREE.Data3DTexture(rgba, size, size, size);
    tex3d.format = THREE.RGBAFormat;
    tex3d.minFilter = tex3d.magFilter = THREE.LinearFilter;
    tex3d.wrapS = tex3d.wrapT = tex3d.wrapR = THREE.ClampToEdgeWrapping;
    tex3d.unpackAlignment = 1;
    tex3d.needsUpdate = true;
    return { tex3d, size };
  }, []);

  const create3DFromImg = useCallback((img) => {
    const size = img.height;
    if (img.width !== size * size) throw new Error(`PNG LUT 크기 오류: width(${img.width}) ≠ height²(${size * size})`);
    const canvas = document.createElement('canvas');
    canvas.width = img.width; canvas.height = img.height;
    const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const rgba = new Uint8Array(size * size * size * 4);
    for (let b = 0; b < size; b++)
      for (let g = 0; g < size; g++)
        for (let r = 0; r < size; r++) {
          const si = (g * canvas.width + (b * size + r)) * 4;
          const di = (b * size * size + g * size + r) * 4;
          rgba[di] = data[si]; rgba[di+1] = data[si+1]; rgba[di+2] = data[si+2]; rgba[di+3] = 255;
        }
    const tex3d = new THREE.Data3DTexture(rgba, size, size, size);
    tex3d.format = THREE.RGBAFormat; tex3d.minFilter = tex3d.magFilter = THREE.LinearFilter;
    tex3d.wrapS = tex3d.wrapT = tex3d.wrapR = THREE.ClampToEdgeWrapping;
    tex3d.unpackAlignment = 1; tex3d.needsUpdate = true;
    return { tex3d, dataUrl: canvas.toDataURL('image/png') };
  }, []);

  const handleLUTUpload = useCallback((e) => {
    const file = e.target.files[0]; if (!file) return;
    const isCube = file.name.toLowerCase().endsWith('.cube');

    if (isCube) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          if (lutRef.current) lutRef.current.dispose();
          const { tex3d, size } = parseCubeLUT(ev.target.result);
          lutRef.current = tex3d; setLut3D(tex3d);
          setParams(p => ({ ...p, lutEnabled: true, lutDataUrl: null, lutName: file.name, lutSize: size }));
        } catch (err) {
          alert('LUT 파싱 오류:\n' + err.message);
        }
      };
      reader.onerror = () => alert('파일 읽기 실패');
      reader.readAsText(file);
    } else {
      const objUrl = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        try {
          if (lutRef.current) lutRef.current.dispose();
          const { tex3d, dataUrl } = create3DFromImg(img);
          lutRef.current = tex3d; setLut3D(tex3d);
          setParams(p => ({ ...p, lutEnabled: true, lutDataUrl: dataUrl, lutName: file.name, lutSize: img.height }));
        } catch (err) {
          alert('PNG LUT 오류:\n' + err.message + '\n\n※ PNG LUT는 width = height² 형식이어야 합니다.');
        } finally { URL.revokeObjectURL(objUrl); }
      };
      img.onerror = () => { URL.revokeObjectURL(objUrl); alert('이미지 로드 실패'); };
      img.src = objUrl;
    }
  }, [parseCubeLUT, create3DFromImg]);

  const handlePatternUpload = useCallback((e) => {
    const file = e.target.files[0]; if (!file) return;
    const url = URL.createObjectURL(file);
    new THREE.TextureLoader().load(url, (tex) => {
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      tex.minFilter = THREE.LinearMipMapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.generateMipmaps = true;
      tex.needsUpdate = true;
      if (patternRef.current) patternRef.current.dispose();
      patternRef.current = tex;
      setPatternTex(tex);
      setParams(p => ({ ...p, patternEnabled: true }));
      URL.revokeObjectURL(url);
    });
  }, []);

  const neutralPatternTex = useMemo(() => {
    const t = new THREE.DataTexture(new Uint8Array([128, 128, 128, 255]), 1, 1, THREE.RGBAFormat);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.needsUpdate = true;
    return t;
  }, []);

  const aspect = texture ? texture.image?.width / texture.image?.height || 1 : 1;

  useEffect(() => {
    const id = 'unarrived-style';
    if (document.getElementById(id)) return;
    const s = document.createElement('style');
    s.id = id;
    s.textContent = `
      .ua-slider { -webkit-appearance: none; appearance: none; height: 3px;
        border-radius: 2px; outline: none; cursor: pointer; width: 100%;
        background: linear-gradient(to right, var(--acc, #4a90d9) var(--val, 50%), #323232 var(--val, 50%)); }
      .ua-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
        width: 12px; height: 12px; border-radius: 50%;
        background: #d0d0d0; border: 1.5px solid #888; cursor: grab; transition: background .1s; }
      .ua-slider::-webkit-slider-thumb:active { background: #fff; cursor: grabbing; }
      .ua-slider::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%;
        background: #d0d0d0; border: 1.5px solid #888; cursor: grab; }
      .ua-num::-webkit-inner-spin-button, .ua-num::-webkit-outer-spin-button { -webkit-appearance: none; }
      .ua-num { -moz-appearance: textfield; }
      .ua-panel::-webkit-scrollbar { width: 4px; }
      .ua-panel::-webkit-scrollbar-track { background: #1a1a1a; }
      .ua-panel::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 2px; }
      .ua-file { font-size: 11px; color: #8a9ab0; cursor: pointer; width: 100%;
        background: #252525; border: 1px dashed #3a3a3a; border-radius: 4px;
        padding: 7px 10px; box-sizing: border-box; transition: border-color .15s; }
      .ua-file:hover { border-color: #4a90d9; color: #b0c4de; }
      .ua-toggle-row:hover { color: #d4d4d4 !important; }
    `;
    document.head.appendChild(s);
  }, []);

  const sidebarContent = (
    <>
      <Panel label="IMAGE">
        <label className="ua-file">
          {/* ✨ RAW 방지: 브라우저가 지원하는 일반 이미지 포맷만 허용되도록 명시 */}
          <input type="file" accept="image/jpeg, image/png, image/webp" onChange={handleImageUpload} style={{ display: 'none' }} />
          ＋ 이미지 불러오기 (JPG / PNG / WebP)
        </label>
      </Panel>

      <Panel label="DECONSTRUCT" accent="#c070ff">
        <EffectRow name="Melting" onReset={() => sp('melting')(0)} tag="픽셀 융해">
          <Sl label="Melting" min={0} max={1} value={params.melting} onChange={sp('melting')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>

        <Sep/>

        <EffectRow name="Crystallize" onReset={() => sp('crystallize')(0)} tag="Voronoi 셀">
          <Sl label="Crystallize" min={0} max={1} value={params.crystallize} onChange={sp('crystallize')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>
        <EffectRow name="Threshold Cascade" onReset={() => sp('threshCascade')(0)} tag="명도대역 분리">
          <Sl label="Threshold Cascade" min={0} max={1} value={params.threshCascade} onChange={sp('threshCascade')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>
        <EffectRow name="Sort Glitch" onReset={() => sp('sortGlitch')(0)} tag="1D 픽셀 정렬">
          <Sl label="Sort Glitch" min={0} max={1} value={params.sortGlitch} onChange={sp('sortGlitch')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>

        <Sep/>

        <EffectRow name="Freq Peel" onReset={() => sp('freqPeel')(0)} tag="주파수 분리">
          <Sl label="Freq Peel" min={0} max={1} value={params.freqPeel} onChange={sp('freqPeel')} accent="#c070ff" compact defaultValue={0}/>
          {params.freqPeel > 0 && (
            <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
              {[['Edge', 0], ['Flat', 1], ['Invert', 2]].map(([l, v]) => (
                <button key={v} onClick={() => sp('freqPeelMode')(v)} style={{
                  flex: 1, padding: '4px 0', fontSize: 10, borderRadius: 3, cursor: 'pointer',
                  background: params.freqPeelMode === v ? '#c070ff' : C.bgDeep,
                  color: params.freqPeelMode === v ? '#fff' : C.textDim,
                  border: `1px solid ${params.freqPeelMode === v ? '#d090ff' : C.line2}`,
                }}>{l}</button>
              ))}
            </div>
          )}
        </EffectRow>

        <Sep/>

        <EffectRow name="Edge Harvest" onReset={() => sp('edgeHarvest')(0)} tag="구조 골격 추출">
          <Sl label="Edge Harvest" min={0} max={1} value={params.edgeHarvest} onChange={sp('edgeHarvest')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>
        <EffectRow name="Ink Spread" onReset={() => sp('inkSpread')(0)} tag="잉크 번짐">
          <Sl label="Ink Spread" min={0} max={1} value={params.inkSpread} onChange={sp('inkSpread')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>
        <EffectRow name="Stipple" onReset={() => sp('stipple')(0)} tag="블루노이즈 점묘화">
          <Sl label="Stipple" min={0} max={1} value={params.stipple} onChange={sp('stipple')} accent="#c070ff" compact defaultValue={0}/>
        </EffectRow>
      </Panel>

      <Panel label="RECONSTRUCT" accent="#50c8a0">
        <div style={{ fontSize: 10, color: C.textMute, marginBottom: 10, lineHeight: 1.5 }}>
          해체된 신호를 인쇄 매체 언어로 재구성합니다.
        </div>
        <EffectRow name="Scatter" onReset={() => sp('scatter')(0)} tag="확률 점묘">
          <Sl label="Scatter" min={0} max={1} value={params.scatter} onChange={sp('scatter')} accent="#50c8a0" compact defaultValue={0}/>
        </EffectRow>
        <Sep/>
        <EffectRow name="Halftone" onReset={() => sp('halftone')(0)} tag="도트 망판">
          <Sl label="Halftone" min={0} max={1} value={params.halftone} onChange={sp('halftone')} accent="#50c8a0" compact defaultValue={0}/>
        </EffectRow>
        {params.halftone > 0 && (
          <div style={{ paddingLeft: 10, borderLeft: '2px solid #50c8a033', marginBottom: 8 }}>
            <Sl label="Scale" min={10} max={400} step={1} value={params.halftoneScale} onChange={sp('halftoneScale')} accent="#50c8a0" defaultValue={40}/>
            <Sl label="Angle °" min={0} max={180} step={1}
              value={Math.round(params.halftoneAngle * (180 / Math.PI))}
              onChange={v => sp('halftoneAngle')(v * (Math.PI / 180))}
              accent="#50c8a0" defaultValue={0}/>
          </div>
        )}
        <Sep/>
        <EffectRow name="Cross-Hatch" onReset={() => sp('crossHatch')(0)} tag="교차 선각">
          <Sl label="Cross-Hatch" min={0} max={1} value={params.crossHatch} onChange={sp('crossHatch')} accent="#50c8a0" compact defaultValue={0}/>
        </EffectRow>
        {params.crossHatch > 0 && (
          <div style={{ paddingLeft: 10, borderLeft: '2px solid #50c8a033', marginBottom: 8 }}>
            <Sl label="Scale" min={10} max={400} step={1} value={params.crossHatchScale} onChange={sp('crossHatchScale')} accent="#50c8a0" defaultValue={40}/>
            <Sl label="Angle °" min={0} max={180} step={1}
              value={Math.round(params.crossHatchAngle * (180 / Math.PI))}
              onChange={v => sp('crossHatchAngle')(v * (Math.PI / 180))}
              accent="#50c8a0" defaultValue={0}/>
          </div>
        )}
      </Panel>

      <Panel label="TEXTURE STAMP" accent="#e8a040">
        <div style={{ fontSize: 10, color: C.textMute, marginBottom: 10, lineHeight: 1.6 }}>
          입력 텍스처가 공간 드라이버로 작동합니다.
          패턴의 채널값이 해체·합성 효과의 <em style={{ color: C.textDim }}>위치·강도·스크린</em>을 결정합니다.
        </div>
        <label className="ua-file" style={{ marginBottom: 10, display: 'block',
          borderColor: params.patternEnabled ? '#e8a04088' : '#3a3a3a' }}>
          <input type="file" accept="image/jpeg, image/png, image/webp" onChange={handlePatternUpload} style={{ display: 'none' }} />
          {patternTex ? '✓  패턴 로드됨 — 재업로드' : '＋  패턴 텍스처 불러오기'}
        </label>
        {patternTex && (
          <>
            <Tog label="Pattern Enabled" checked={params.patternEnabled} onChange={sp('patternEnabled')} />
            {params.patternEnabled && (
              <>
                <FieldLabel>Mode</FieldLabel>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 10 }}>
                  {[
                    ['Displace', 0, '높이맵 UV 변위'],
                    ['Mask',     1, '효과 공간 게이팅'],
                    ['Threshold',2, '디더 스크린 교체'],
                    ['Overlay',  3, 'Soft Light 합성'],
                  ].map(([label, v, hint]) => (
                    <button key={v} onClick={() => sp('patternMode')(v)} style={{
                      padding: '6px 4px', fontSize: 10, borderRadius: 3, cursor: 'pointer',
                      background: params.patternMode === v ? '#e8a040' : C.bgDeep,
                      color: params.patternMode === v ? '#000' : C.textDim,
                      border: `1px solid ${params.patternMode === v ? '#f0b860' : C.line2}`,
                      lineHeight: 1.3, textAlign: 'center',
                    }}>
                      <div style={{ fontWeight: '600' }}>{label}</div>
                      <div style={{ fontSize: 9, opacity: 0.7, marginTop: 1 }}>{hint}</div>
                    </button>
                  ))}
                </div>
                <FieldLabel>Source Channel</FieldLabel>
                <div style={{ display: 'flex', gap: 4, marginBottom: 10 }}>
                  {[['L', 0], ['R', 1], ['G', 2], ['B', 3], ['~L', 4]].map(([l, v]) => (
                    <button key={v} onClick={() => sp('patternChannel')(v)} style={{
                      flex: 1, padding: '5px 0', fontSize: 10, borderRadius: 3, cursor: 'pointer',
                      background: params.patternChannel === v ? '#e8a040' : C.bgDeep,
                      color: params.patternChannel === v ? '#000' : C.textDim,
                      border: `1px solid ${params.patternChannel === v ? '#f0b860' : C.line2}`,
                      fontWeight: params.patternChannel === v ? '700' : '400',
                    }}>{l}</button>
                  ))}
                </div>
                <Sep/>
                <Sl label="Intensity"  min={0}   max={1}   value={params.patternIntensity} onChange={sp('patternIntensity')} accent="#e8a040" defaultValue={0.7}/>
                <Sl label="Scale"      min={0.5} max={20}  value={params.patternScale}     onChange={sp('patternScale')}     accent="#e8a040" defaultValue={4.0}/>
                <Sl label="Angle °"    min={0}   max={360} step={1}
                  value={Math.round(params.patternAngle * (180 / Math.PI))}
                  onChange={v => sp('patternAngle')(v * (Math.PI / 180))}
                  accent="#e8a040" defaultValue={0}/>
                <div style={{ marginTop: 8, padding: '7px 10px', background: '#1c1a14',
                  borderRadius: 4, border: '1px solid #3a3010', fontSize: 10, color: '#9a8060', lineHeight: 1.6 }}>
                  {params.patternMode === 0 && '밝은 부분이 앞으로 튀어나옵니다. 어두운 부분은 당겨집니다.'}
                  {params.patternMode === 1 && '밝은 부분에서 Deconstruct 효과가 강하게 적용됩니다.'}
                  {params.patternMode === 2 && '패턴이 블루노이즈를 대체합니다. 임계값 텍스처로 직접 작동합니다.'}
                  {params.patternMode === 3 && '50% 회색 = 변화 없음. 밝으면 밝게, 어두우면 어둡게.'}
                </div>
              </>
            )}
          </>
        )}
      </Panel>

      <Panel label="MODE">
        <div style={{ display: 'flex', gap: 6 }}>
          <ModeBtn active={!params.colorMode} onClick={() => sp('colorMode')(false)}>B &amp; W</ModeBtn>
          <ModeBtn active={params.colorMode}  onClick={() => sp('colorMode')(true)}>COLOR</ModeBtn>
        </div>
      </Panel>

      {params.colorMode && (
        <>
          <Panel label="WHITE BALANCE">
            <Row label="Temperature" hint="← Cool · Warm →" hintColor="#5a8fc0">
              <Sl label="Temperature" min={-1} max={1} value={params.colorTemp} onChange={sp('colorTemp')} defaultValue={0.0}/>
            </Row>
            <Row label="Tint" hint="← Green · Magenta →" hintColor="#8a7a9a">
              <Sl label="Tint" min={-1} max={1} value={params.colorTint} onChange={sp('colorTint')} defaultValue={0.0}/>
            </Row>
          </Panel>
          <Panel label="COLOR GRADING">
            <Sl label="Saturation" min={0} max={2.5} value={params.saturation} onChange={sp('saturation')}  defaultValue={1.0}/>
            <Sep />
            <FieldLabel>Split Toning <Muted>— 회색 = 중립</Muted></FieldLabel>
            <div style={{ display: 'flex', gap: 8, margin: '6px 0 8px' }}>
              <TintSwatch label="Shadows" value={rgbOffsetToHex(params.shadowTint)}
                onChange={e => sp('shadowTint')(hexToRgbOffset(e.target.value))}
                onReset={() => sp('shadowTint')({ ...TINT_NEUTRAL })} />
              <TintSwatch label="Highlights" value={rgbOffsetToHex(params.highlightTint)}
                onChange={e => sp('highlightTint')(hexToRgbOffset(e.target.value))}
                onReset={() => sp('highlightTint')({ ...TINT_NEUTRAL })} />
            </div>
            <Sl label="Split Balance" min={-1} max={1} value={params.splitBalance} onChange={sp('splitBalance')}  defaultValue={0.0}/>
          </Panel>
          <Panel label="ASC CDL">
            <FieldLabel>Lift <Muted>암부 오프셋</Muted></FieldLabel>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
              <Sl label="R" min={-0.2} max={0.2} value={params.lift.r} onChange={v => setParams(p => ({...p, lift: {...p.lift, r: v}}))} accent="#c05050" defaultValue={0}/>
              <Sl label="G" min={-0.2} max={0.2} value={params.lift.g} onChange={v => setParams(p => ({...p, lift: {...p.lift, g: v}}))} accent="#50a050" defaultValue={0}/>
              <Sl label="B" min={-0.2} max={0.2} value={params.lift.b} onChange={v => setParams(p => ({...p, lift: {...p.lift, b: v}}))} accent="#4a7abf" defaultValue={0}/>
            </div>
            <FieldLabel>Gain <Muted>명부 배율</Muted></FieldLabel>
            <div style={{ display: 'flex', gap: 6 }}>
              <Sl label="R" min={0.5} max={2.0} value={params.gain.r} onChange={v => setParams(p => ({...p, gain: {...p.gain, r: v}}))} accent="#c05050" defaultValue={1}/>
              <Sl label="G" min={0.5} max={2.0} value={params.gain.g} onChange={v => setParams(p => ({...p, gain: {...p.gain, g: v}}))} accent="#50a050" defaultValue={1}/>
              <Sl label="B" min={0.5} max={2.0} value={params.gain.b} onChange={v => setParams(p => ({...p, gain: {...p.gain, b: v}}))} accent="#4a7abf" defaultValue={1}/>
            </div>
          </Panel>
        </>
      )}

      {!params.colorMode && (
        <Panel label="DITHER COLORS">
          <div style={{ display: 'flex', gap: 8 }}>
            <TintSwatch label="Dark tone" value={params.color1}
              onChange={e => sp('color1')(e.target.value)} noReset />
            <TintSwatch label="Light tone" value={params.color2}
              onChange={e => sp('color2')(e.target.value)} noReset />
          </div>
        </Panel>
      )}

      <Panel label="TONE MAPPING">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <Muted>Reinhard ←</Muted><Muted>→ ACES</Muted>
        </div>
        <Sl label="Curve Mix" min={0} max={1} value={params.acesMix} onChange={sp('acesMix')}  defaultValue={0.4}/>
        <Sep />
        {/* ✨ 문구 수정: RAW 오해 방지 */}
        <Tog label="Linear Input" sub="Log / HDR 렌더링 이미지일 때만 ON" checked={params.linearize} onChange={sp('linearize')} />
      </Panel>

      <Panel label="EXPOSURE">
        <Sl label="Shadow Recovery" min={0}   max={1.0} value={params.shadows}    onChange={sp('shadows')}     defaultValue={0.2}/>
        <Sl label="Highlights"      min={0.5} max={2.0} value={params.highlights} onChange={sp('highlights')}  defaultValue={1.0}/>
        <Sl label="Gamma"           min={0.5} max={2.0} value={params.gamma}      onChange={sp('gamma')}       defaultValue={1.0}/>
        <Sl label="Contrast"        min={0.1} max={3.0} value={params.contrast}   onChange={sp('contrast')}    defaultValue={1.0}/>
        <Sl label="Brightness"      min={-0.5} max={0.5} value={params.brightness} onChange={sp('brightness')}  defaultValue={0.0}/>
      </Panel>

      <Panel label="DITHERING">
        <Tog label="Enable Dithering" sub="디더링 및 포스터라이즈 온/오프" checked={params.ditherEnabled} onChange={sp('ditherEnabled')} />
        
        {params.ditherEnabled && (
          <div style={{ marginTop: 8, paddingLeft: 10, borderLeft: '2px solid #333' }}>
            <Sl label="Posterize Levels"  min={2}   max={32}  step={1}     value={params.levels}      onChange={sp('levels')}        defaultValue={4} />
            <Sl label="Edge Softness"     min={0}   max={1.0} step={0.005} value={params.softness}    onChange={sp('softness')}      defaultValue={0.1} />
            <Sep />
            <Sl label="Pattern Scale"     min={0.5} max={300} step={0.5}   value={params.density}     onChange={sp('density')}       defaultValue={30} />
            <Sl label="Dither Amount"     min={0}   max={2.0} step={0.01}  value={params.noiseAmount} onChange={sp('noiseAmount')}   defaultValue={0} />
            <Sl label="Bayer ↔ Blue Mix" min={0} max={1.0} step={0.01} value={params.superposition} onChange={sp('superposition')} defaultValue={0.0} 
                sub="0 = Blue Noise(확률론적) / 1 = Bayer(규칙적 망점)"/>
          </div>
        )}
      </Panel>

      <Panel label="TEMPORAL" accent="#5a4aaa">
        <Tog label="Animate Effects" sub="끄면 화면이 스스로 안정화되며 정지" checked={params.animate} onChange={sp('animate')} />
        
        {params.animate && (
          <div style={{ paddingLeft: 10, borderLeft: '2px solid #5a4aaa33', marginTop: 8, marginBottom: 8 }}>
            <EffectRow name="Glitch Mosh" onReset={() => sp('glitch')(0)} tag="글리치">
              <Sl label="Glitch Mosh" min={0} max={1} value={params.glitch} onChange={sp('glitch')} accent="#5a4aaa" compact defaultValue={0}/>
            </EffectRow>
            <EffectRow name="Fluidity" onReset={() => sp('fluidity')(0)} tag="유체화">
              <Sl label="Fluidity" min={0} max={1} value={params.fluidity} onChange={sp('fluidity')} accent="#5a4aaa" compact defaultValue={0}/>
            </EffectRow>
            <EffectRow name="Scan Displacement" onReset={() => sp('scanDisplace')(0)} tag="스캔라인 부패">
              <Sl label="Scan Displacement" min={0} max={1} value={params.scanDisplace} onChange={sp('scanDisplace')} accent="#5a4aaa" compact defaultValue={0}/>
            </EffectRow>
          </div>
        )}

        <Sep />
        <Tog
          label="Temporal Accumulation"
          sub={params.temporalEnabled ? 'ON: 노이즈 누적 안티앨리어싱' : 'OFF: 단일 프레임 출력'}
          checked={params.temporalEnabled}
          onChange={sp('temporalEnabled')}
        />
        {params.temporalEnabled && (
          <Sl label="Accum. Strength" min={0} max={0.99} step={0.01} value={params.accumulation} onChange={sp('accumulation')} defaultValue={0.85}
              sub="과거 프레임 보존율 (높을수록 부드럽지만 잔상 증가)"/>
        )}
      </Panel>

      <Panel label="OPTICS &amp; FILM">
        <Tog label="Sharpen" checked={params.sharpen} onChange={sp('sharpen')} />
        {params.sharpen && <Sl label="Amount" min={0} max={1.5} value={params.sharpenAmount} onChange={sp('sharpenAmount')}  defaultValue={0.5}/>}
        <Tog label="Vignette" checked={params.vignette} onChange={sp('vignette')} />
        {params.vignette && <Sl label="Strength" min={0} max={1.5} value={params.vignetteStrength} onChange={sp('vignetteStrength')}  defaultValue={0.5}/>}
        <Tog label="Anamorphic Flare" checked={params.flare} onChange={sp('flare')} />
        {params.flare && (
          <>
            <Sl label="Threshold"  min={0.7}  max={1.0}  value={params.flareThresh}  onChange={sp('flareThresh')}  defaultValue={0.9}/>
            <Sl label="Amount"     min={0}    max={2.0}  value={params.flareAmount}  onChange={sp('flareAmount')}  defaultValue={0.5}/>
            <Sl label="Width"      min={0.001} max={0.08} value={params.flareWidth}   onChange={sp('flareWidth')}   defaultValue={0.01}/>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: 11, color: C.text, flex: 1, userSelect: 'none' }}>Color</span>
              <div style={{ position: 'relative', width: 46, height: 22 }}>
                <div style={{
                  position: 'absolute', inset: 0, borderRadius: 3,
                  background: params.flareColor, border: `1px solid ${C.line2}`, pointerEvents: 'none',
                }} />
                <input type="color" value={params.flareColor} onChange={e => sp('flareColor')(e.target.value)}
                  style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%', height: '100%', border: 'none', padding: 0 }} />
              </div>
            </div>
          </>
        )}
        <Tog label="Spectral CA" checked={params.caEnable} onChange={sp('caEnable')} />
        {params.caEnable && (
          <>
            <select value={params.caType} onChange={e => setParams(p => ({...p, caType: parseInt(e.target.value)}))}
              style={{ width: '100%', marginBottom: 6, fontSize: 11, background: '#252525', color: '#c8c8c8',
                border: '1px solid #3d3d3d', padding: '5px 8px', borderRadius: 3, outline: 'none' }}>
              <option value={0}>Radial</option>
              <option value={1}>Edge Only</option>
            </select>
            <Sl label="CA Amount" min={0} max={0.06} value={params.caAmount} onChange={sp('caAmount')}  defaultValue={0.015}/>
          </>
        )}
        <Sep />
        <Sl label="Film Halation" min={0} max={1.0} value={params.halation} onChange={sp('halation')}  defaultValue={0.0}/>
        {params.halation > 0 && (
          <Sl label="Halation Thresh" min={0.3} max={0.95} value={params.halationThresh} onChange={sp('halationThresh')} defaultValue={0.65}/>
        )}
        <Sl label="Film Grain"    min={0} max={0.3}  value={params.grain}     onChange={sp('grain')}       defaultValue={0.03}/>
        <Sl label="Grain Size"    min={1} max={8}    value={params.grainSize} onChange={sp('grainSize')}   defaultValue={2.0}/>
      </Panel>

      <Panel label="LUT &amp; PRESETS">
        {!webgl2 && <div style={{ background: '#3a1a1a', border: '1px solid #a03030', borderRadius: 4, padding: '6px 10px', marginBottom: 8, fontSize: 11, color: '#e07070' }}>⚠ WebGL2 미지원 — LUT 비활성</div>}

        <label className="ua-file" style={{
          marginBottom: 8, display: 'block',
          opacity: webgl2 ? 1 : 0.4, pointerEvents: webgl2 ? 'auto' : 'none',
          borderColor: params.lutEnabled ? '#4a90d988' : '#3a3a3a',
        }}>
          <input type="file" accept=".cube,.png,image/png"
            onChange={handleLUTUpload} style={{ display: 'none' }} disabled={!webgl2} />
          {params.lutName
            ? `✓  ${params.lutName}${params.lutSize ? ` (${params.lutSize}³)` : ''} — 재업로드`
            : '＋  LUT 불러오기  (.cube / PNG)'}
        </label>

        {params.lutEnabled && !params.lutDataUrl && params.lutName && (
          <div style={{ fontSize: 10, color: '#6a8050', background: '#1a1e14',
            border: '1px solid #3a4020', borderRadius: 4, padding: '5px 8px', marginBottom: 8, lineHeight: 1.5 }}>
            ℹ .cube LUT는 프리셋에 저장되지 않습니다.<br/>프리셋 복원 후 LUT를 다시 업로드해주세요.
          </div>
        )}

        {params.lutEnabled && <Sl label="LUT Mix" min={0} max={1.0} value={params.lutMix} onChange={sp('lutMix')} defaultValue={1.0}/>}

        {params.lutEnabled && (
          <button onClick={() => {
            if (lutRef.current) { lutRef.current.dispose(); lutRef.current = null; }
            setLut3D(null);
            setParams(p => ({ ...p, lutEnabled: false, lutDataUrl: null, lutName: null, lutSize: null }));
          }} style={{
            width: '100%', marginBottom: 8, padding: '5px 0', fontSize: 10,
            background: 'none', border: '1px solid #4a2020', color: '#8a5050',
            borderRadius: 3, cursor: 'pointer', fontFamily: 'inherit',
          }}>✕ LUT 제거</button>
        )}
        <Sep />
        <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
          <input placeholder="프리셋 이름" value={presetName} onChange={e => setPresetName(e.target.value)}
            style={{ flex: 1, padding: '5px 8px', background: '#252525', color: '#c8c8c8',
              border: '1px solid #3d3d3d', fontSize: 11, borderRadius: 3, outline: 'none',
              fontFamily: 'inherit' }} />
          <button onClick={() => {
            try {
              const u = [...presets, { id: Date.now(), name: presetName || 'Preset', params }];
              localStorage.setItem(PRESETS_KEY, JSON.stringify(u)); setPresets(u); setPresetName('');
            } catch { alert('저장 용량 초과!'); }
          }} style={{ padding: '5px 12px', background: '#4a90d9', color: '#fff', border: 'none',
            cursor: 'pointer', fontWeight: '600', fontSize: 11, borderRadius: 3, whiteSpace: 'nowrap' }}>
            저장
          </button>
        </div>
        <div style={{ maxHeight: 110, overflowY: 'auto', borderRadius: 3, border: '1px solid #2d2d2d', background: '#1c1c1c' }}>
          {presets.length === 0 && <div style={{ padding: '10px 12px', fontSize: 11, color: '#555', fontStyle: 'italic' }}>저장된 프리셋 없음</div>}
          {presets.map(p => (
            <div key={p.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '7px 12px', borderBottom: '1px solid #252525', cursor: 'pointer',
              transition: 'background .1s' }}
              onMouseEnter={e => e.currentTarget.style.background = '#252525'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <span onClick={() => {
                setParams(prev => ({ ...prev, ...p.params }));
                if (p.params.lutDataUrl) {
                  const i = new Image();
                  i.onload = () => {
                    try {
                      const { tex3d } = create3DFromImg(i);
                      if (lutRef.current) lutRef.current.dispose();
                      lutRef.current = tex3d; setLut3D(tex3d);
                    } catch {}
                  };
                  i.src = p.params.lutDataUrl;
                } else if (p.params.lutEnabled) {
                  setLut3D(null);
                  setParams(prev => ({ ...prev, ...p.params, lutEnabled: false }));
                }
              }} style={{ fontSize: 12, color: '#b8b8b8', fontWeight: '500', flex: 1 }}>{p.name}</span>
              <span onClick={() => {
                const u = presets.filter(x => x.id !== p.id); setPresets(u); localStorage.setItem(PRESETS_KEY, JSON.stringify(u));
              }} style={{ color: '#804040', fontSize: 11, padding: '2px 6px', borderRadius: 2,
                background: '#2a1a1a', cursor: 'pointer', marginLeft: 8 }}>✕</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel label="EXPORT">
        <Sl label="Long Edge (px)" min={1000} max={8000} step={100} value={params.resolution} onChange={sp('resolution')} defaultValue={4000}/>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: -4, marginBottom: 8 }}>
          <span style={{ fontSize: 11, color: '#5a7a9a', fontVariantNumeric: 'tabular-nums' }}>
            {aspect >= 1
              ? `${params.resolution} × ${Math.round(params.resolution / aspect)} px`
              : `${Math.round(params.resolution * aspect)} × ${params.resolution} px`}
          </span>
        </div>
      </Panel>

      {/* ✨ 추출 버튼 텍스트에서 혼동을 줄 수 있는 RAW 단어 삭제 */}
      <button
        onClick={() => !params.isExportingFlag && saveRef.current?.()}
        style={{
          width: '100%', padding: '13px 0', marginTop: 4,
          background: params.isExportingFlag
            ? '#252525'
            : 'linear-gradient(180deg, #4a90d9 0%, #2d6db0 100%)',
          color: params.isExportingFlag ? '#555' : '#fff',
          border: params.isExportingFlag ? '1px solid #333' : '1px solid #5aa0e9',
          borderRadius: 5, fontWeight: '600', cursor: params.isExportingFlag ? 'not-allowed' : 'pointer',
          fontSize: 12, letterSpacing: '0.06em', transition: 'opacity .15s',
        }}
      >
        {params.isExportingFlag ? '⏳  추출 중…' : '↓  EXTRACT HIGH-RES MOMENT'}
      </button>
      <div style={{ height: 20 }} />
    </>
  );

  return (
    <div style={{ width: '100vw', height: '100dvh', display: 'flex',
      background: '#141414', color: '#c8c8c8',
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      position: 'relative', overflow: 'hidden' }}>

      {isMobile && (
        <button onClick={() => setSidebarOpen(o => !o)}
          style={{
            position: 'absolute', top: 12, left: 12, zIndex: 200,
            width: 36, height: 36, background: '#2d2d2d', border: '1px solid #3d3d3d',
            borderRadius: 5, cursor: 'pointer',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4,
          }}
          aria-label="메뉴"
        >
          {[0, 1, 2].map(i => (
            <span key={i} style={{
              display: 'block', width: 16, height: 1.5, background: '#c8c8c8',
              transition: 'transform 0.2s, opacity 0.2s',
              transform: sidebarOpen
                ? i === 0 ? 'rotate(45deg) translate(4px, 4px)'
                : i === 2 ? 'rotate(-45deg) translate(4px, -4px)' : 'scaleX(0)'
                : 'none',
              opacity: sidebarOpen && i === 1 ? 0 : 1,
            }} />
          ))}
        </button>
      )}

      {/* ── 사이드바 ── */}
      <div className="ua-panel" style={{
        width: 272, flexShrink: 0,
        background: '#1e1e1e',
        borderRight: '1px solid #2d2d2d',
        zIndex: 100,
        display: 'flex', flexDirection: 'column',
        ...(isMobile ? {
          position: 'absolute', top: 0, left: 0, bottom: 0,
          width: '82vw', maxWidth: 300,
          transform: sidebarOpen ? 'translateX(0)' : 'translateX(-100%)',
          transition: 'transform 0.22s cubic-bezier(.4,0,.2,1)',
          boxShadow: sidebarOpen ? '6px 0 32px rgba(0,0,0,0.6)' : 'none',
        } : {}),
      }}>
        {/* 헤더 */}
        <div style={{
          padding: '14px 14px 12px',
          borderBottom: '1px solid #2d2d2d',
          marginLeft: isMobile ? 44 : 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ fontSize: 13, fontWeight: '700', color: '#e0e0e0', letterSpacing: '-0.3px' }}>
              The Unarrived Engine
            </div>
            <button
              onClick={() => {
                if (window.confirm('모든 설정을 초기 상태로 되돌리시겠습니까?')) {
                  setParams({ ...INITIAL_PARAMS });
                }
              }}
              title="모든 파라미터를 초기값으로 리셋"
              style={{
                fontSize: 10, fontWeight: '600', letterSpacing: '0.04em',
                background: 'none', border: '1px solid #3a3a3a',
                color: '#6a8aaa', borderRadius: 4,
                padding: '3px 8px', cursor: 'pointer',
                transition: 'border-color 0.15s, color 0.15s',
              }}
              onMouseEnter={e => { e.target.style.borderColor = '#5a7a9a'; e.target.style.color = '#8aaabb'; }}
              onMouseLeave={e => { e.target.style.borderColor = '#3a3a3a'; e.target.style.color = '#6a8aaa'; }}
            >RESET ALL</button>
          </div>
          <div style={{ fontSize: 10, color: '#5a7a9a', marginTop: 2, letterSpacing: '0.02em' }}>
            Linear · ACES · Reinhard · Color Dither
          </div>
        </div>

        <div style={{ flex: 1, padding: '6px 0 8px', overflowY: 'auto' }} className="ua-panel">
          {sidebarContent}
        </div>
      </div>

      {isMobile && sidebarOpen && (
        <div onClick={() => setSidebarOpen(false)}
          style={{ position: 'absolute', inset: 0, zIndex: 90, background: 'rgba(0,0,0,0.5)' }} />
      )}

      {/* ── 캔버스 ── */}
      <div
        ref={canvasContainerRef}
        style={{
          flex: 1, position: 'relative', background: '#0a0a0a', minWidth: 0,
          width: isMobile ? '100%' : undefined,
          overflow: 'hidden',
          cursor: zoom > 1 ? (isPanningState ? 'grabbing' : 'grab') : 'default',
          userSelect: 'none',
        }}
        onMouseDown={e => {
          if (zoom <= 1) return;
          isPanning.current = true;
          setIsPanningState(true);
          panStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
        }}
        onMouseMove={e => {
          if (!isPanning.current) return;
          const np = clampPan(
            panStart.current.px + (e.clientX - panStart.current.x),
            panStart.current.py + (e.clientY - panStart.current.y),
            zoom
          );
          setPan(np);
        }}
        onMouseUp={() => { isPanning.current = false; setIsPanningState(false); }}
        onMouseLeave={() => { isPanning.current = false; setIsPanningState(false); }}
        onTouchStart={e => {
          if (zoom <= 1 || e.touches.length !== 1) return;
          isPanning.current = true;
          panStart.current = { x: e.touches[0].clientX, y: e.touches[0].clientY, px: pan.x, py: pan.y };
        }}
        onTouchMove={e => {
          if (!isPanning.current || e.touches.length !== 1) return;
          e.preventDefault();
          const np = clampPan(
            panStart.current.px + (e.touches[0].clientX - panStart.current.x),
            panStart.current.py + (e.touches[0].clientY - panStart.current.y),
            zoom
          );
          setPan(np);
        }}
        onTouchEnd={() => { isPanning.current = false; }}
        onWheel={e => {
          e.preventDefault();
          setZoomVisible(true);
          clearTimeout(zoomHideTimer.current);
          zoomHideTimer.current = setTimeout(() => setZoomVisible(false), 2000);
          const LEVELS = [1, 2, 4];
          const cur = LEVELS.indexOf(zoom);
          if (e.deltaY < 0 && cur < LEVELS.length - 1) {
            const next = LEVELS[cur + 1];
            setZoom(next);
            setPan(p => clampPan(p.x, p.y, next));
          } else if (e.deltaY > 0 && cur > 0) {
            const next = LEVELS[cur - 1];
            setZoom(next);
            if (next === 1) setPan({ x: 0, y: 0 });
            else setPan(p => clampPan(p.x, p.y, next));
          }
        }}
      >
        <div style={{
          width: '100%', height: '100%',
          transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          transformOrigin: 'center center',
          transition: isPanningState ? 'none' : 'transform 0.18s cubic-bezier(0.22,1,0.36,1)',
          willChange: 'transform',
        }}>
        <Canvas
          camera={{ position: [0, 0, 10], fov: 43.6028 }}
          gl={{ antialias: true, preserveDrawingBuffer: true, outputColorSpace: THREE.LinearSRGBColorSpace }}
        >
          <Suspense fallback={null}>
            {texture && bNoise && (
              <EngineCore
                texture={texture} bNoise={bNoise} lut3D={lut3D || defaultLut3D}
                patternTex={patternTex || neutralPatternTex}
                params={params} aspect={aspect}
                saveRef={saveRef}
                setIsExporting={setIsExporting}
                setExportResult={setExportResult}
              />
            )}
          </Suspense>
        </Canvas>
        {!texture && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', pointerEvents: 'none', gap: 10 }}>
            <div style={{ fontSize: 28, opacity: 0.12 }}>▣</div>
            <div style={{ fontSize: 12, color: '#333' }}>이미지를 사이드바에서 불러오세요</div>
          </div>
        )}
        </div>

        <div style={{
          position: 'absolute', bottom: 14, right: 14, zIndex: 50,
          display: 'flex', flexDirection: 'column', gap: 4,
          opacity: zoomVisible ? 1 : 0,
          pointerEvents: zoomVisible ? 'auto' : 'none',
          transition: 'opacity 0.3s ease',
        }}>
          {[4, 2, 1].map(lvl => (
            <button key={lvl}
              onClick={() => {
                setZoom(lvl);
                if (lvl === 1) setPan({ x: 0, y: 0 });
                else setPan(p => clampPan(p.x, p.y, lvl));
              }}
              title={lvl === 1 ? '원본 크기' : `${lvl}배 확대`}
              style={{
                width: 36, height: 28, fontSize: 11, fontWeight: '700',
                fontFamily: 'inherit', letterSpacing: '0.03em',
                borderRadius: 4, cursor: 'pointer', border: 'none',
                background: zoom === lvl ? '#4a90d9' : 'rgba(20,20,20,0.82)',
                color: zoom === lvl ? '#fff' : '#7a9ab8',
                backdropFilter: 'blur(8px)',
                boxShadow: zoom === lvl ? '0 0 0 1px #4a90d9' : '0 0 0 1px rgba(80,100,130,0.3)',
                transition: 'all 0.12s',
              }}
            >{lvl === 1 ? '1×' : `${lvl}×`}</button>
          ))}
          {zoom > 1 && (pan.x !== 0 || pan.y !== 0) && (
            <button
              onClick={() => setPan({ x: 0, y: 0 })}
              title="중앙으로"
              style={{
                width: 36, height: 24, fontSize: 14, borderRadius: 4,
                cursor: 'pointer', border: 'none',
                background: 'rgba(20,20,20,0.82)',
                color: '#7a9ab8',
                backdropFilter: 'blur(8px)',
                boxShadow: '0 0 0 1px rgba(80,100,130,0.3)',
                transition: 'all 0.12s', marginTop: 2,
              }}
            >⊕</button>
          )}
        </div>

        {zoom > 1 && (
          <div style={{
            position: 'absolute', bottom: 14, left: 14, zIndex: 50,
            fontSize: 10, color: '#4a90d9', letterSpacing: '0.08em',
            background: 'rgba(10,10,10,0.75)', padding: '3px 7px',
            borderRadius: 3, backdropFilter: 'blur(6px)',
            border: '1px solid rgba(74,144,217,0.3)',
            pointerEvents: 'none',
            opacity: zoomVisible ? 1 : 0,
            transition: 'opacity 0.3s ease',
          }}>
            {zoom}× — 드래그로 이동
          </div>
        )}
      </div>

      {exportResult && (
        <div onClick={() => setExportResult(null)}
          style={{
            position: 'fixed', inset: 0, zIndex: 999,
            background: 'rgba(0,0,0,0.82)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
          }}
        >
          <div onClick={e => e.stopPropagation()}
            style={{
              background: '#232323', border: '1px solid #383838',
              borderRadius: 8, padding: 20, width: '100%', maxWidth: 520, maxHeight: '90vh',
              display: 'flex', flexDirection: 'column', gap: 14, overflowY: 'auto',
              boxShadow: '0 24px 64px rgba(0,0,0,0.7)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: '700', color: '#e8e8e8' }}>
                  추출 완료 — {exportResult.w} × {exportResult.h} px
                </div>
              </div>
              <button onClick={() => setExportResult(null)}
                style={{ background: '#333', border: '1px solid #444', color: '#aaa',
                  cursor: 'pointer', fontSize: 14, width: 28, height: 28,
                  borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                ✕
              </button>
            </div>

            <img src={exportResult.dataUrl} alt="export"
              style={{ width: '100%', maxHeight: '55vh', objectFit: 'contain',
                borderRadius: 4, border: '1px solid #2d2d2d', background: '#111' }} />

            <a href={exportResult.dataUrl}
              download={`The_Unarrived_${exportResult.w}x${exportResult.h}.png`}
              style={{
                display: 'block', padding: '11px 0', textAlign: 'center',
                background: 'linear-gradient(180deg,#4a90d9,#2d6db0)',
                color: '#fff', fontWeight: '600', fontSize: 12,
                borderRadius: 5, textDecoration: 'none', letterSpacing: '0.04em',
                border: '1px solid #5aa0e9',
              }}
            >
              ↓  PNG 다운로드
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================
// 6. UI 컴포넌트 — Photoshop / Lightroom 스타일
// =============================================================

const C = {
  bg:       '#1e1e1e',
  bgDeep:   '#181818',
  panel:    '#252525',
  line:     '#2d2d2d',
  line2:    '#333333',
  text:     '#c8c8c8',
  textDim:  '#8a8a8a',
  textMute: '#555555',
  accent:   '#4a90d9',
  accentHi: '#6aaae9',
};

const Panel = ({ label, children, accent }) => {
  const [open, setOpen] = useState(true);
  return (
    <div style={{ borderBottom: `1px solid ${C.line}` }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '7px 14px', background: 'none', border: 'none', cursor: 'pointer',
          color: accent ?? C.textDim,
        }}
      >
        <span style={{ fontSize: 10, fontWeight: '700', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {label}
        </span>
        <span style={{
          fontSize: 8, color: C.textMute,
          transform: open ? 'rotate(0deg)' : 'rotate(-90deg)',
          transition: 'transform 0.15s',
          display: 'inline-block',
        }}>▼</span>
      </button>
      {open && (
        <div style={{ padding: '4px 14px 12px' }}>
          {children}
        </div>
      )}
    </div>
  );
};

const Sl = ({ label, sub, min, max, step = 0.001, value, onChange, accent, defaultValue, compact }) => {
  const pct = ((value - min) / (max - min) * 100).toFixed(1) + '%';
  const color = accent ?? C.accent;
  const disp = step >= 1 ? Math.round(value) : parseFloat(value).toFixed(step < 0.01 ? 3 : 2);
  const handleReset = defaultValue !== undefined
    ? () => onChange(Math.min(max, Math.max(min, defaultValue)))
    : null;

  if (compact) {
    return (
      <div style={{ marginBottom: 6, minWidth: 0, display: 'flex', alignItems: 'center', gap: 6 }}>
        <input
          type="range" min={min} max={max} step={step} value={value}
          className="ua-slider"
          style={{ '--val': pct, '--acc': color, flex: 1 }}
          onChange={e => onChange(parseFloat(e.target.value))}
        />
        <input
          type="number" value={disp} step={step}
          className="ua-num"
          onClick={handleReset ?? undefined}
          title={handleReset ? `클릭 → 초기값 (${step >= 1 ? Math.round(defaultValue) : defaultValue})` : undefined}
          onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(Math.min(max, Math.max(min, v))); }}
          style={{
            width: 46, background: C.bgDeep, color: C.text,
            border: `1px solid ${C.line2}`, fontSize: 11,
            textAlign: 'right', padding: '2px 5px', borderRadius: 3,
            outline: 'none', flexShrink: 0, fontFamily: 'inherit',
          }}
        />
      </div>
    );
  }

  return (
    <div style={{ marginBottom: 8, minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
        <span
          onDoubleClick={handleReset ?? undefined}
          title={handleReset ? `더블클릭 → 초기값 (${step >= 1 ? Math.round(defaultValue) : defaultValue})` : undefined}
          style={{
            fontSize: 11, color: C.text, letterSpacing: '-0.1px',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            flex: 1, marginRight: 6,
            cursor: handleReset ? 'pointer' : 'default',
            userSelect: 'none',
          }}
        >{label}</span>
        <input
          type="number" value={disp} step={step}
          className="ua-num"
          onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(Math.min(max, Math.max(min, v))); }}
          style={{
            width: 46, background: C.bgDeep, color: C.text,
            border: `1px solid ${C.line2}`, fontSize: 11,
            textAlign: 'right', padding: '2px 5px', borderRadius: 3,
            outline: 'none', flexShrink: 0, fontFamily: 'inherit',
          }}
        />
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        className="ua-slider"
        style={{ '--val': pct, '--acc': color }}
        onChange={e => onChange(parseFloat(e.target.value))}
      />
      {sub && <div style={{ fontSize: 9.5, color: C.textDim, marginTop: 2, lineHeight: 1.4, opacity: 0.7 }}>{sub}</div>}
    </div>
  );
};

const Tog = ({ label, sub, checked, onChange }) => (
  <label className="ua-toggle-row" style={{
    display: 'flex', alignItems: 'flex-start', gap: 8, cursor: 'pointer',
    marginBottom: 8, userSelect: 'none', color: checked ? C.text : C.textDim,
  }}>
    <span style={{
      flexShrink: 0, marginTop: 1,
      width: 14, height: 14, borderRadius: 3,
      background: checked ? C.accent : C.bgDeep,
      border: `1px solid ${checked ? C.accentHi : C.line2}`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      transition: 'background .12s, border-color .12s',
    }}>
      {checked && <span style={{ color: '#fff', fontSize: 9, lineHeight: 1 }}>✓</span>}
    </span>
    <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} style={{ display: 'none' }} />
    <div>
      <div style={{ fontSize: 11, fontWeight: '500', lineHeight: 1.3 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: C.textMute, marginTop: 1 }}>{sub}</div>}
    </div>
  </label>
);

const ModeBtn = ({ active, onClick, children }) => (
  <button onClick={onClick} style={{
    flex: 1, padding: '7px 0',
    background: active ? C.accent : C.bgDeep,
    color: active ? '#fff' : C.textDim,
    border: `1px solid ${active ? C.accentHi : C.line2}`,
    borderRadius: 4, fontWeight: '600', fontSize: 11,
    cursor: 'pointer', letterSpacing: '0.04em', transition: 'all 0.12s',
  }}>
    {children}
  </button>
);

const TintSwatch = ({ label, value, onChange, onReset, noReset }) => (
  <div style={{ flex: 1 }}>
    <div style={{ fontSize: 10, color: C.textDim, marginBottom: 4 }}>{label}</div>
    <div style={{ position: 'relative', height: 28 }}>
      <div style={{
        position: 'absolute', inset: 0,
        borderRadius: 4, background: value,
        border: `1px solid ${C.line2}`, pointerEvents: 'none',
      }} />
      <input type="color" value={value} onChange={onChange} style={{
        position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer',
        width: '100%', height: '100%', border: 'none', padding: 0,
      }} />
    </div>
    {!noReset && (
      <button onClick={onReset} style={{
        width: '100%', marginTop: 4, padding: '3px 0',
        fontSize: 10, background: C.bgDeep, color: C.textDim,
        border: `1px solid ${C.line}`, cursor: 'pointer', borderRadius: 3,
        fontFamily: 'inherit',
      }}>Reset</button>
    )}
  </div>
);

const Row = ({ hint, hintColor, children }) => (
  <div>
    {hint && (
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9,
        color: hintColor ?? C.textMute, marginBottom: 2, letterSpacing: '0.02em' }}>
        {hint.split('·').map((t, i) => <span key={i}>{t.trim()}</span>)}
      </div>
    )}
    {children}
  </div>
);

const Muted = ({ children }) => (
  <span style={{ fontSize: 10, color: C.textMute, fontWeight: '400' }}>{children}</span>
);
const FieldLabel = ({ children }) => (
  <div style={{ fontSize: 10, color: C.textDim, fontWeight: '600', letterSpacing: '0.04em',
    textTransform: 'uppercase', marginBottom: 5, marginTop: 2 }}>{children}</div>
);
const Sep = () => (
  <div style={{ height: 1, background: C.line, margin: '8px 0' }} />
);

const EffectRow = ({ name, tag, children, onReset }) => (
  <div style={{ marginBottom: 10 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 3 }}>
      <span
        onDoubleClick={onReset ?? undefined}
        title={onReset ? '더블클릭 → 초기값으로 리셋' : undefined}
        style={{
          fontSize: 11, color: C.text, fontWeight: '600',
          cursor: onReset ? 'pointer' : 'default',
          userSelect: 'none',
        }}
      >{name}</span>
      <span style={{ fontSize: 9, color: C.textMute, letterSpacing: '0.04em' }}>{tag}</span>
    </div>
    {children}
  </div>
);