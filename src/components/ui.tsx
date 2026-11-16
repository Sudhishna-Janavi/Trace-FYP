/**
 * The pieces every screen is built from.
 *
 * Each one mirrors a class in `trace.css`, so a card here has the same
 * radius, the same hairline border and the same tinted surface as a card
 * there. Keeping them in one file means the resemblance is maintained in one
 * place rather than re-derived on every screen.
 */
import React from 'react';
import {
  ActivityIndicator, Pressable, StyleSheet, Text, TextInput, TextStyle,
  View, ViewStyle,
} from 'react-native';
import { SvgXml } from 'react-native-svg';

import { useTheme } from '../ThemeContext';
import {
  Palette, layerColour, layerSoft, money as formatMoney, radius, space, type,
} from '../theme';

/* ------------------------------------------------------------------ text -- */

type TextTone = 'ink' | 'muted' | 'faint' | 'accent' | 'income' | 'spend'
  | 'review' | 'onAccent';

export function T({ children, variant = 'body', tone = 'ink', style, numberOfLines }: {
  children: React.ReactNode;
  variant?: keyof typeof type;
  tone?: TextTone;
  style?: TextStyle | TextStyle[];
  numberOfLines?: number;
}) {
  const { p } = useTheme();
  return (
    <Text
      numberOfLines={numberOfLines}
      style={[type[variant] as TextStyle, { color: p[tone] }, style]}>
      {children}
    </Text>
  );
}

/* ------------------------------------------------------------------ card -- */

export function Card({ children, style, tinted = false, padded = true }: {
  children: React.ReactNode;
  style?: ViewStyle | ViewStyle[];
  tinted?: boolean;
  padded?: boolean;
}) {
  const { p } = useTheme();
  return (
    <View style={[{
      backgroundColor: tinted ? p.surface2 : p.surface,
      borderColor: p.line,
      borderWidth: StyleSheet.hairlineWidth,
      borderRadius: radius.large,
      padding: padded ? space.lg : 0,
      overflow: 'hidden',
    }, style]}>
      {children}
    </View>
  );
}

export function SectionTitle({ children, note }: {
  children: React.ReactNode;
  note?: string;
}) {
  return (
    <View style={{ marginBottom: space.md, marginTop: space.xl }}>
      <T variant="heading">{children}</T>
      {note ? <T variant="small" tone="muted" style={{ marginTop: 3 }}>{note}</T> : null}
    </View>
  );
}

/* ------------------------------------------------------------------ pill -- */

export function Pill({ label, tone = 'neutral', icon }: {
  label: string;
  tone?: 'neutral' | 'accent' | 'review' | 'spend';
  icon?: string;
}) {
  const { p } = useTheme();
  const colours: Record<string, [string, string]> = {
    neutral: [p.surface2, p.muted],
    accent: [p.accentSoft, p.accent],
    review: [p.reviewSoft, p.review],
    spend: [p.dangerSoft, p.danger],
  };
  const [background, foreground] = colours[tone];
  return (
    <View style={{
      backgroundColor: background,
      borderRadius: radius.pill,
      paddingHorizontal: 9,
      paddingVertical: 4,
      alignSelf: 'flex-start',
    }}>
      <Text style={[type.label as TextStyle, { color: foreground }]}>
        {icon ? `${icon} ` : ''}{label}
      </Text>
    </View>
  );
}

/**
 * Which layer decided, and whether its reason is exact.
 *
 * The distinction is the project's argument in miniature, so it is never
 * implied by colour alone: the word Exact or Approximate is always present.
 */
export function LayerChip({ layer, label, exact }: {
  layer: string;
  label: string;
  exact?: boolean;
}) {
  const { p } = useTheme();
  const colour = layerColour(layer, p);
  return (
    <View style={{ flexDirection: 'row', gap: 6, alignItems: 'center' }}>
      {exact !== undefined ? (
        <View style={{
          backgroundColor: layerSoft(layer, p),
          borderRadius: radius.pill,
          paddingHorizontal: 9,
          paddingVertical: 4,
        }}>
          <Text style={[type.label as TextStyle, { color: colour }]}>
            {exact ? '✓ Exact' : '⚠ Approximate'}
          </Text>
        </View>
      ) : null}
      <View style={{
        backgroundColor: p.surface2,
        borderRadius: radius.pill,
        paddingHorizontal: 9,
        paddingVertical: 4,
      }}>
        <Text style={[type.label as TextStyle, { color: p.muted }]}>{label}</Text>
      </View>
    </View>
  );
}

/* ----------------------------------------------------------------- money -- */

export function Money({ value, variant = 'body', decimals = 2, signed = true }: {
  value: number;
  variant?: keyof typeof type;
  decimals?: 0 | 2;
  signed?: boolean;
}) {
  const { p } = useTheme();
  const colour = !signed ? p.ink : value > 0 ? p.income : value < 0 ? p.spend : p.ink;
  return (
    <Text style={[type[variant] as TextStyle, {
      color: colour, fontVariant: ['tabular-nums'],
    }]}>
      {formatMoney(value, decimals)}
    </Text>
  );
}

/* --------------------------------------------------------------- buttons -- */

export function Button({ label, onPress, kind = 'primary', disabled, busy, style }: {
  label: string;
  onPress: () => void;
  kind?: 'primary' | 'ghost' | 'quiet';
  disabled?: boolean;
  busy?: boolean;
  style?: ViewStyle;
}) {
  const { p } = useTheme();
  const primary = kind === 'primary';
  const background = primary ? p.accent : kind === 'quiet' ? 'transparent' : p.surface;
  const foreground = primary ? p.onAccent : p.ink;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled: !!disabled || !!busy }}
      disabled={disabled || busy}
      onPress={onPress}
      style={({ pressed }) => [{
        backgroundColor: background,
        borderColor: primary || kind === 'quiet' ? 'transparent' : p.line,
        borderWidth: primary || kind === 'quiet' ? 0 : StyleSheet.hairlineWidth,
        borderRadius: radius.card,
        // 46pt: the web build uses the same minimum, because a thumb is not a
        // mouse pointer and anything smaller is missed as often as hit.
        minHeight: 46,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: space.lg,
        opacity: pressed || disabled ? 0.65 : 1,
      }, style]}>
      {busy ? <ActivityIndicator color={foreground} />
            : <Text style={[type.heading as TextStyle, { color: foreground, fontSize: 15 }]}>
                {label}
              </Text>}
    </Pressable>
  );
}

/* ----------------------------------------------------------------- field -- */

export function Field({ label, value, onChangeText, ...rest }: {
  label: string;
  value: string;
  onChangeText: (next: string) => void;
} & React.ComponentProps<typeof TextInput>) {
  const { p } = useTheme();
  return (
    <View style={{ marginTop: space.lg }}>
      <Text style={[type.label as TextStyle, { color: p.muted, marginBottom: 6 }]}>
        {label}
      </Text>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholderTextColor={p.faint}
        style={[type.body as TextStyle, {
          color: p.ink,
          backgroundColor: p.surface,
          borderColor: p.line,
          borderWidth: StyleSheet.hairlineWidth,
          borderRadius: radius.card,
          paddingHorizontal: 13,
          paddingVertical: 12,
          minHeight: 46,
        }]}
        {...rest}
      />
    </View>
  );
}

/* ---------------------------------------------------------------- states -- */

export function Loading({ note }: { note?: string }) {
  const { p } = useTheme();
  return (
    <View style={{ paddingVertical: 48, alignItems: 'center', gap: space.md }}>
      <ActivityIndicator color={p.accent} />
      {note ? <T variant="small" tone="muted">{note}</T> : null}
    </View>
  );
}

export function Empty({ title, body, action }: {
  title: string;
  body?: string;
  action?: React.ReactNode;
}) {
  return (
    <Card tinted style={{ alignItems: 'flex-start', gap: space.sm }}>
      <T variant="heading">{title}</T>
      {body ? <T variant="small" tone="muted">{body}</T> : null}
      {action}
    </Card>
  );
}

export function Problem({ message, onRetry }: { message: string; onRetry?: () => void }) {
  const { p } = useTheme();
  return (
    <View style={{
      backgroundColor: p.dangerSoft,
      borderRadius: radius.card,
      padding: space.lg,
      gap: space.md,
    }}>
      <Text style={[type.small as TextStyle, { color: p.danger }]}>{message}</Text>
      {onRetry ? <Button label="Try again" kind="ghost" onPress={onRetry} /> : null}
    </View>
  );
}

/* ------------------------------------------------------- server-drawn SVG -- */

/**
 * A chart the server drew, displayed natively.
 *
 * Trace renders its charts as inline SVG on the server so that they follow
 * the page's custom properties and need no charting library. Those same
 * strings are handed to the phone: the chart here is the chart there, not a
 * second implementation that might disagree with it.
 *
 * Two adjustments are needed. The marks reference CSS custom properties,
 * which exist in a browser and not in `react-native-svg`, so each `var(--x)`
 * is resolved against the active palette — which is also what makes the chart
 * follow dark mode. And `<title>` elements, which a browser surfaces as a
 * tooltip, have no meaning under a finger and are removed.
 */
const TOKENS: Record<string, keyof Palette> = {
  '--line': 'line', '--faint': 'faint', '--muted': 'muted', '--ink': 'ink',
  '--bg': 'bg', '--surface': 'surface', '--surface-2': 'surface2',
  '--accent': 'accent', '--accent-soft': 'accentSoft', '--on-accent': 'onAccent',
  '--income': 'income', '--spend': 'spend', '--review': 'review',
  '--review-soft': 'reviewSoft', '--danger': 'danger', '--danger-soft': 'dangerSoft',
};

export function resolveSvg(xml: string, p: Palette): string {
  return xml
    .replace(/var\((--[a-z0-9-]+)\)/g, (whole, token: string) => {
      const key = TOKENS[token];
      return key ? p[key] : whole;
    })
    .replace(/<title>[\s\S]*?<\/title>/g, '')
    .replace(/\sstyle="[^"]*"/g, '');
}

export function ServerSvg({ xml, width }: { xml: string; width: number }) {
  const { p } = useTheme();
  if (!xml) return null;
  const resolved = resolveSvg(xml, p);
  const box = /viewBox="0 0 ([\d.]+) ([\d.]+)"/.exec(resolved);
  const ratio = box ? Number(box[1]) / Number(box[2]) : 3;
  return (
    <SvgXml xml={resolved} width={width} height={width / ratio} />
  );
}

/* ------------------------------------------------------------------ rows -- */

export function Divider() {
  const { p } = useTheme();
  return <View style={{ height: StyleSheet.hairlineWidth, backgroundColor: p.line }} />;
}

export function Row({ children, onPress, style }: {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
}) {
  const { p } = useTheme();
  const content = (
    <View style={[{
      paddingVertical: 13,
      paddingHorizontal: space.lg,
      minHeight: 46,
      justifyContent: 'center',
    }, style]}>
      {children}
    </View>
  );
  if (!onPress) return content;
  return (
    <Pressable
      onPress={onPress}
      android_ripple={{ color: p.surface2 }}
      style={({ pressed }) => ({ backgroundColor: pressed ? p.surface2 : 'transparent' })}>
      {content}
    </Pressable>
  );
}

export { space, radius, type };
