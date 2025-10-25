# Design Guidelines: AI Personality Analysis Platform

## Design Approach
**Selected System:** Medical/Clinical Interface Design (inspired by Apple HIG + healthcare applications)
**Rationale:** Professional psychological analysis demands trust, clarity, and accessibility. Clinical aesthetic emphasizes credibility while maintaining modern usability.

**Core Principles:**
- Hierarchy through typography weight, not color
- Generous whitespace for cognitive breathing room
- Precision over decoration
- Data-forward presentation

## Typography System

**Font Family:** Inter (primary), SF Mono (code/data display)
- Hero/H1: 3xl-4xl, font-bold, tracking-tight
- Section Headers/H2: 2xl, font-semibold
- Analysis Labels/H3: lg, font-medium
- Body Text: base, font-normal, leading-relaxed
- Data/Results: base, font-medium
- Metadata/Captions: sm, text-gray-600

## Layout System

**Spacing Primitives:** Tailwind units of 2, 4, 6, 8, 12, 16, 24
- Component padding: p-6 to p-8
- Section spacing: space-y-8 to space-y-12
- Card margins: m-4
- Button spacing: px-6 py-3

**Grid Structure:**
- Main container: max-w-7xl mx-auto px-6
- Analysis output: max-w-4xl (optimal reading width for clinical text)
- Dashboard cards: grid-cols-1 md:grid-cols-2 lg:grid-cols-3

## Component Library

### Navigation
**Top Bar:**
- Fixed header with subtle border-bottom
- Logo left, navigation center, user profile right
- Height: h-16
- Navigation links: font-medium, spacing mx-8
- Minimal hover: subtle underline offset

### Hero Section (Landing/Marketing)
**Layout:**
- Full-width section with professional imagery
- Two-column: Left (60%) headline + CTA, Right (40%) supporting visual
- min-h-[600px] with centered vertical alignment
- Overlay buttons: backdrop-blur-sm bg-white/10 for glass effect

**Content:**
- H1: "Unlock Deep Personality Insights Through AI"
- Subheading: Explain analysis types (photo/video/text)
- Primary CTA: "Start Analysis" + Secondary: "View Sample Report"

### Analysis Interface (Core Application)

**Action Bar (Critical Requirement):**
Positioned directly above analysis output container:
- Flex row with justify-between
- Left: Analysis title/timestamp
- Right: Button group with gap-3
- Copy Button: "Copy Analysis" with document-duplicate icon
- Delete Button: "Delete" with trash icon, red text on hover
- Both buttons: px-4 py-2, rounded-lg, border styling

**Analysis Output Container:**
- White card with subtle shadow: shadow-sm
- Rounded: rounded-xl
- Padding: p-8
- Border: border border-gray-200
- Background for sections: bg-gray-50 with p-6 rounded-lg

**Results Display Structure:**
1. Analysis Header: Type badge, confidence score, timestamp
2. Key Findings: Grid of metric cards (grid-cols-2 gap-4)
3. Detailed Breakdown: Accordion sections with expand/collapse
4. Visual Data: Progress bars for traits (0-100 scale)
5. Source Reference: Small text linking to input type

### Cards (Features/Dashboard)
**Feature Cards:**
- Aspect ratio: aspect-[4/3]
- Icon top: w-12 h-12, rounded-full bg-gray-100
- Title: text-lg font-semibold, mb-2
- Description: text-gray-600, leading-relaxed
- Hover: subtle lift with shadow-md transition

**Metric Cards:**
- Compact square: p-6
- Large number: text-3xl font-bold
- Label: text-sm text-gray-500
- Border-left accent for categorization

### Forms (Upload/Analysis Input)
**Upload Interface:**
- Drag-and-drop zone: dashed border, rounded-lg, p-12
- Icon: upload-cloud, centered
- File type indicators: small badges
- Progress bar: h-2 rounded-full with animated fill

**Input Fields:**
- Text areas: min-h-[200px] for text analysis
- Labels: font-medium mb-2, above input
- Helper text: text-sm text-gray-500 mt-1
- Focus state: ring-2 ring-offset-2

### Data Visualization
**Personality Trait Bars:**
- Container: space-y-4
- Each trait: Label (font-medium) + bar + score
- Bar: h-3 rounded-full bg-gray-200
- Fill: rounded-full with width percentage
- Score: text-sm font-semibold aligned right

**Confidence Indicators:**
- Circular percentage: 64px diameter
- Ring stroke with completion arc
- Center: percentage value, font-bold

### Modals/Overlays
**Confirmation Dialogs (Delete action):**
- Centered modal: max-w-md
- Icon: warning triangle, text-red-500
- Title: text-xl font-semibold
- Actions: Cancel (ghost) + Confirm Delete (red, solid)

## Images

**Hero Section Image:**
- Professional photograph of diverse people or abstract brain/neural network visualization
- Position: Right 40% of hero, subtle fade effect on left edge
- Style: Modern, clean, slightly desaturated for professional tone
- Aspect ratio: 3:4 portrait orientation

**No images in analysis interface** - maintain clinical focus on data

## Layout Patterns

**Marketing Page Structure:**
1. Hero with image (described above)
2. Features Grid: 3-column showcase (Photo/Video/Text analysis)
3. How It Works: 4-step process with numbered cards
4. Sample Analysis Preview: Embedded example with blur overlay + CTA
5. Trust Section: 2-column (credentials left, testimonials right)
6. Final CTA: Centered with trial information

**Application Layout:**
- Sidebar navigation (w-64, fixed)
- Main content area (flex-1, ml-64)
- Analysis workspace: centered max-w-4xl
- Action bar always visible above output

## Critical Details

**Accessibility:**
- All interactive elements: min 44x44px touch targets
- ARIA labels on icon-only buttons
- Focus indicators: ring-2 with high contrast
- Form validation: inline error messages

**Responsive Breakpoints:**
- Mobile: Stack all columns, full-width cards
- Tablet (md:): 2-column grids
- Desktop (lg:): Full 3-column layouts
- Action buttons: Horizontal on desktop, stacked on mobile

**Professional Polish:**
- Consistent border-radius: rounded-lg (0.5rem) for cards, rounded-md for buttons
- Shadows: Use sparingly - shadow-sm for cards, shadow-md for elevated states
- Transitions: duration-200 ease-in-out for all interactive states
- Loading states: Skeleton screens with pulse animation for data fetching