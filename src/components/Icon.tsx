/**
 * The tab bar's icons.
 *
 * These are the same path data the installed web application draws in
 * `templates/app.html`. Copying the paths rather than reaching for an icon
 * set keeps the two tab bars identical, and keeps the application free of a
 * dependency whose house style is not Trace's.
 *
 * Each is a stroked outline on a 24-unit grid, so the stroke width is the
 * same whatever size it is drawn at.
 */
import React from 'react';
import Svg, { Path } from 'react-native-svg';

export const ICONS = {
  dashboard: 'M3 10.5 12 3l9 7.5M5.5 9.5V20h13V9.5',
  transactions: 'M4 7h16M4 12h16M4 17h10',
  review: 'M12 8v5M12 16.5v.5M10.3 3.6 2.6 17a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 3.6a2 2 0 0 0-3.4 0Z',
  insights: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  accounts: 'M3 9.5 12 4l9 5.5M4.5 9.5V19h15V9.5M3 19h18',
  how: 'M12 17v.5M9.8 9.2a2.3 2.3 0 1 1 3.2 2.1c-.7.3-1 .9-1 1.6v.6M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z',
  more: 'M5 12h.01M12 12h.01M19 12h.01',
  back: 'M15 5l-7 7 7 7',
  close: 'M6 6l12 12M18 6L6 18',
} as const;

export type IconName = keyof typeof ICONS;

export function Icon({ name, colour, size = 23 }: {
  name: IconName;
  colour: string;
  size?: number;
}) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path
        d={ICONS[name]}
        stroke={colour}
        strokeWidth={1.7}
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </Svg>
  );
}
