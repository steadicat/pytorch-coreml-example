import * as React from 'react';
import * as ReactDOM from 'react-dom';
import {piecewiseRegression, piecewiseRegressionWithSplits} from './regression';
import * as dataJSON from '../data.json';
import * as prediction from '../prediction.json';

type Point = [number, number];

export function findRange(values: number[]): [number, number] {
  return values.reduce<[number, number]>(
    ([min, max], value) => [Math.min(min, value), Math.max(max, value)],
    [Infinity, -Infinity],
  );
}

export function makeScale([start, end]: [number, number], [min, max]: [number, number]) {
  return (n: number) => start + (end - start) * (n - min) / (max - min);
}

let data: Array<{points: Point[]; splits: number[]}> = dataJSON;
let useNN = true;
const history: Array<typeof data> = [];

function addExample() {
  history.push(data);
  data = [...data, {points: [], splits: []}];
  useNN = false;
  render();
}

function undo() {
  if (!history.length) return;
  data = history.pop();
  render();
}

function switchToPlain() {
  useNN = false;
  render();
}

function switchToNN() {
  if (history.length > 0) {
    alert(
      'Running the Neural Network on dynamically added data is not implemented yet. Undo or refresh to see NN output on preset data.',
    );
    return;
  }
  useNN = true;
  render();
}

const Button = ({onClick, children, style = {} as React.CSSProperties}) => (
  <button onClick={onClick} style={{marginRight: 8, ...style}}>
    {children}
  </button>
);

function predict(points: Point[]) {
  return [];
}

function findTrendLines(points, useNN = false, exampleID: number = null) {
  if (useNN) {
    const splits = prediction[exampleID] || [];
    return piecewiseRegressionWithSplits(points, splits);
  } else {
    return piecewiseRegression(points);
  }
}

function coordinatesToSplits(xs: number[], points: Point[]): number[] {
  if (xs.length === 0) return [];
  const splits = [];
  let nextX = 0;
  for (let i = 0; i < points.length; i++) {
    if (points[i][0] > xs[nextX]) {
      splits.push(i - 1);
      nextX++;
      if (nextX >= xs.length) break;
    }
  }
  return splits;
}

type TrendLine = {start: number; end: number; slope: number; y: number};

function linesToSplits(lines: TrendLine[], points: Point[]): number[] {
  if (lines.length === 0) return [];
  const splits = new Set<number>();
  let nextLine = 0;
  let inLine = false;
  for (let i = 0; i < points.length; i++) {
    if (points[i][0] === lines[nextLine].start) {
      splits.add(i - 1);
      inLine = true;
    } else if (points[i][0] === lines[nextLine].end) {
      splits.add(i);
      inLine = false;
      nextLine++;
      if (nextLine >= lines.length) break;
    } else if (!inLine) {
      splits.add(i - 1);
      splits.add(i);
    }
  }
  splits.delete(-1);
  splits.delete(points.length - 1);
  const res = [...splits];
  res.sort((a, b) => a - b);
  return res;
}

const App = ({
  examples,
  width,
  height,
  useNN = false,
}: {
  examples: Array<{points: Point[]; splits: number[]}>;
  width: number;
  height: number;
  useNN?: boolean;
}) => (
  <div style={{margin: 20, marginTop: 50}}>
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        padding: 10,
        background: 'white',
        borderBottom: '1px solid #eee',
        display: 'flex',
      }}>
      <Button onClick={addExample}>Add Example</Button>
      <div style={{font: 'caption', marginLeft: 'auto'}}>
        {useNN ? 'Using Neural Network' : 'Using Plain Math'}
      </div>
      <Button
        onClick={useNN ? switchToPlain : switchToNN}
        style={{marginLeft: 16, marginRight: 'auto'}}>
        {useNN ? 'Use Plain Math' : 'Use Neural Network'}
      </Button>
      <Button onClick={undo} style={{visibility: history.length > 0 ? null : 'hidden'}}>
        Undo
      </Button>
    </div>
    {examples.map(({points, splits}, i) => {
      const xRange: Point = [0, width]; // || findRange(points.map(p => p[0]));
      const yRange: Point = [0, height]; // || findRange(points.map(p => p[1]));
      const xScale = makeScale([0, width], xRange);
      const yScale = makeScale([height, 0], yRange);

      const xScaleInverse = makeScale(xRange, [0, width]);
      const yScaleInverse = makeScale(yRange, [height, 0]);
      const lines = findTrendLines(points, useNN, i);

      splits.sort((a, b) => a - b);
      const correct =
        coordinatesToSplits(splits, points).join(',') === linesToSplits(lines, points).join(',');

      const addPoint = ({nativeEvent: {offsetX, offsetY}}: React.MouseEvent<SVGElement>) => {
        history.push(JSON.parse(JSON.stringify(data)));
        points.push([xScaleInverse(offsetX), yScaleInverse(offsetY)]);
        points.sort((a, b) => a[0] - b[0]);
        switchToPlain();
      };
      const addSplit = (event: React.MouseEvent<SVGElement>) => {
        event.stopPropagation();
        history.push(JSON.parse(JSON.stringify(data)));
        splits.push(xScaleInverse(event.nativeEvent.offsetX));
        splits.sort((a, b) => a[0] - b[0]);
        switchToPlain();
      };

      return (
        <svg
          key={i}
          width={width}
          height={height}
          onClick={addPoint}
          style={{
            border: correct ? '1px solid #eee' : '3px solid #f00',
            boxSizing: 'border-box',
            display: 'block',
            margin: '20px 0',
          }}>
          <rect x={0} y={0} width={width} height={10} fill="#eee" onClick={addSplit} />
          {splits.map((x, i) => (
            <line key={i} stroke="#ccc" x1={xScale(x)} y1={0} x2={xScale(x)} y2={height} />
          ))}
          {points.map(([x, y], i) => (
            <circle key={i} fill="#01beff" r={5} cx={xScale(x)} cy={yScale(y)} />
          ))}
          {lines.map(({slope, y, start, end}, i) => (
            <line
              key={i}
              stroke="#f00"
              x1={xScale(start)}
              y1={yScale(slope * start + y)}
              x2={xScale(end)}
              y2={yScale(slope * end + y)}
            />
          ))}
        </svg>
      );
    })}
  </div>
);

const app = document.getElementById('app');

function render() {
  ReactDOM.render(<App examples={data} width={800} height={200} useNN={useNN} />, app);
}

render();
