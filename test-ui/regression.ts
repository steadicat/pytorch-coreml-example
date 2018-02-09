export type Point = [number, number];

export function regression(coords: Point[]) {
  let sumOfX = 0;
  let sumOfY = 0;
  let sumOfXX = 0;
  let sumOfXY = 0;
  for (const [x, y] of coords) {
    sumOfX += x;
    sumOfY += y;
    sumOfXX += x * x;
    sumOfXY += x * y;
  }

  const len = coords.length;
  const run = len * sumOfXX - sumOfX * sumOfX;
  const rise = len * sumOfXY - sumOfX * sumOfY;
  const slope = run === 0 ? 0 : rise / run;
  return {slope, y: sumOfY / len - slope * sumOfX / len};
}

function mse({slope, y: y0}: {slope: number; y: number}, coords: Point[]) {
  let error = 0;
  const weights = coords.map((p, i) => {
    const prev = coords[i - 1];
    const next = coords[i + 1];
    if (!prev && !next) return 1;
    if (!prev) return next[0] - p[0];
    if (!next) return p[0] - prev[0];
    return Math.min(p[0] - prev[0], next[0] - p[0]);
  });

  coords.forEach(([x, y], i) => {
    error += Math.pow(x * slope + y0 - y, 2) * weights[i];
  });
  return error / weights.reduce((a, b) => a + b);
}

type TrendLine = {
  slope: number;
  y: number;
  start: number;
  end: number;
};

export function piecewiseRegressionWithSplits(
  coords: Point[],
  splits: number[] | null,
): TrendLine[] {
  if (!splits) return [];
  const parts = [];
  let start = 0;
  for (const split of splits) {
    parts.push(coords.slice(start, split + 1));
    start = split + 1;
  }
  if (start < coords.length - 1) {
    parts.push(coords.slice(start));
  }
  return parts.filter(part => part.length > 1).map(part => {
    const {slope, y} = regression(part);
    return {slope, y, start: part[0][0], end: part[part.length - 1][0]};
  });
}

const minGainPercentage = 0.1;

export function piecewiseRegression(coords: Point[], minGain: number | null = null): TrendLine[] {
  if (coords.length <= 1) return [];
  if (coords.length <= 3)
    return [
      {
        ...regression(coords),
        start: coords[0][0],
        end: coords[coords.length - 1][0],
      },
    ];
  const originalLine = regression(coords);
  const originalLoss = mse(originalLine, coords);
  minGain || (minGain = originalLoss * minGainPercentage);
  const a = [...coords];
  const b: typeof a = [];
  b.unshift(a.pop()!);
  b.unshift(a.pop()!);
  let bestSplit = 1;
  let bestGain = 0;
  for (; a.length > 1; b.unshift(a.pop()!)) {
    const aLine = regression(a);
    const aLoss = mse(aLine, a);
    const bLine = regression(b);
    const bLoss = mse(bLine, b);
    const gain = originalLoss - (aLoss + bLoss) / 2;
    if (gain > bestGain) {
      bestGain = gain;
      bestSplit = a.length;
    }
  }
  if (bestGain > minGain) {
    return [
      ...piecewiseRegression(coords.slice(0, bestSplit), minGain),
      ...piecewiseRegression(coords.slice(bestSplit), minGain),
    ];
  } else {
    return [
      {
        ...regression(coords),
        start: coords[0][0],
        end: coords[coords.length - 1][0],
      },
    ];
  }
}
