import Markdown from "react-markdown";

const components = {
  a({ href, children, ...props }) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
        {children}
      </a>
    );
  },
};

export default function MarkdownContent({ children }) {
  return <Markdown components={components}>{children}</Markdown>;
}
